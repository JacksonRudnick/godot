#ifndef MODULE_FRAMEGEN_H
#define MODULE_FRAMEGEN_H

#include "core/io/image.h"
#include "scene/main/node.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <condition_variable>
#include <mutex>
#include <thread>

class Framegen : public Node {
	GDCLASS(Framegen, Node);

private:
	static constexpr int INPUT_WIDTH = 640;
	static constexpr int INPUT_HEIGHT = 360;
	static constexpr int INPUT_CHANNELS = 3;
	static constexpr int PLAYER_INPUT_FEATURES = 18;

	torch::jit::script::Module module;
	bool module_loaded = false;
	torch::Device device = torch::kCPU;
	torch::ScalarType inference_dtype = torch::kFloat32;
	torch::Tensor input_tensor;
	torch::Tensor player_input_tensor;
	torch::Tensor input_staging_u8;
	torch::Tensor player_input_staging;
	Vector<uint8_t> output_buffer;
	std::vector<torch::jit::IValue> forward_inputs;
	std::thread worker_thread;
	std::condition_variable worker_cv;
	mutable std::mutex worker_mutex;
	std::mutex inference_mutex;
	bool worker_running = false;
	bool worker_stop_requested = false;
	bool worker_has_job = false;
	bool worker_has_ready_frame = false;
	Ref<Image> worker_pending_frame;
	Dictionary worker_pending_input;
	Ref<Image> worker_ready_frame;

	void _ensure_static_buffers();
	Ref<Image> _run_inference(const Ref<Image> &f_t, Dictionary inp_t);
	void _worker_loop();
	void _start_worker();
	void _stop_worker();
	static void _publish_present_frame(const Ref<Image> &p_img);
	static std::mutex present_frame_mutex;
	static Ref<Image> latest_present_frame;

	torch::Tensor _process_player_inputs(Dictionary inp_t);
	torch::Tensor _process_input_frame(const Ref<Image> &f_t);

protected:
	static void _bind_methods();

public:
	~Framegen();

	bool load_module(const String &p_path);
	Ref<Image> generate_frame(const Ref<Image> &f_t, Dictionary inp_t);
	bool submit_frame(const Ref<Image> &f_t, Dictionary inp_t);
	bool is_generated_frame_ready() const;
	Ref<Image> consume_generated_frame();
	static bool consume_latest_present_frame(Ref<Image> &r_img);
};

#endif // MODULE_FRAMEGEN_H