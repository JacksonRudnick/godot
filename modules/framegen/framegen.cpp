#include "framegen.h"

#include "core/object/class_db.h"
#include <cstring>

std::mutex Framegen::present_frame_mutex;
Ref<Image> Framegen::latest_present_frame;

Framegen::~Framegen() {
	_stop_worker();
}

void Framegen::_ensure_static_buffers() {
	if (!input_staging_u8.defined()) {
		input_staging_u8 = torch::empty({ INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS }, torch::dtype(torch::kUInt8).device(torch::kCPU));
	}

	if (!player_input_staging.defined()) {
		player_input_staging = torch::zeros({ 1, PLAYER_INPUT_FEATURES }, torch::dtype(torch::kFloat32).device(torch::kCPU));
	}

	if (!input_tensor.defined() || input_tensor.device() != device) {
		input_tensor = torch::empty({ 1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH }, torch::dtype(inference_dtype).device(device));
	}

	if (!player_input_tensor.defined() || player_input_tensor.device() != device) {
		player_input_tensor = torch::empty({ 1, PLAYER_INPUT_FEATURES }, torch::dtype(inference_dtype).device(device));
	}

	if (input_tensor.scalar_type() != inference_dtype) {
		input_tensor = input_tensor.to(inference_dtype);
	}

	if (player_input_tensor.scalar_type() != inference_dtype) {
		player_input_tensor = player_input_tensor.to(inference_dtype);
	}

	forward_inputs.clear();
	forward_inputs.reserve(2);
}

void Framegen::_publish_present_frame(const Ref<Image> &p_img) {
	if (p_img.is_null()) {
		return;
	}

	std::lock_guard<std::mutex> lock(present_frame_mutex);
	latest_present_frame = p_img;
}

bool Framegen::consume_latest_present_frame(Ref<Image> &r_img) {
	std::lock_guard<std::mutex> lock(present_frame_mutex);
	if (latest_present_frame.is_null()) {
		r_img.unref();
		return false;
	}

	r_img = latest_present_frame;
	latest_present_frame.unref();
	return true;
}

void Framegen::_start_worker() {
	std::lock_guard<std::mutex> lock(worker_mutex);
	if (worker_running) {
		return;
	}

	worker_stop_requested = false;
	worker_has_job = false;
	worker_has_ready_frame = false;
	worker_running = true;
	worker_thread = std::thread(&Framegen::_worker_loop, this);
}

void Framegen::_stop_worker() {
	{
		std::lock_guard<std::mutex> lock(worker_mutex);
		if (!worker_running) {
			return;
		}
		worker_stop_requested = true;
	}
	worker_cv.notify_all();

	if (worker_thread.joinable()) {
		worker_thread.join();
	}

	std::lock_guard<std::mutex> lock(worker_mutex);
	worker_running = false;
	worker_has_job = false;
	worker_has_ready_frame = false;
	worker_pending_frame.unref();
	worker_ready_frame.unref();
}

void Framegen::_worker_loop() {
	while (true) {
		Ref<Image> job_frame;
		Dictionary job_input;

		{
			std::unique_lock<std::mutex> lock(worker_mutex);
			worker_cv.wait(lock, [this]() { return worker_stop_requested || worker_has_job; });

			if (worker_stop_requested) {
				break;
			}

			job_frame = worker_pending_frame;
			job_input = worker_pending_input;
			worker_has_job = false;
		}

		if (job_frame.is_null()) {
			continue;
		}

		Ref<Image> generated;
		{
			std::lock_guard<std::mutex> infer_lock(inference_mutex);
			generated = _run_inference(job_frame, job_input);
		}

		std::lock_guard<std::mutex> lock(worker_mutex);
		worker_ready_frame = generated;
		worker_has_ready_frame = generated.is_valid();
	}
}

void Framegen::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load_module", "path"), &Framegen::load_module);
	ClassDB::bind_method(D_METHOD("generate_frame", "f_t", "inp_t"), &Framegen::generate_frame);
	ClassDB::bind_method(D_METHOD("submit_frame", "f_t", "inp_t"), &Framegen::submit_frame);
	ClassDB::bind_method(D_METHOD("is_generated_frame_ready"), &Framegen::is_generated_frame_ready);
	ClassDB::bind_method(D_METHOD("consume_generated_frame"), &Framegen::consume_generated_frame);
}

bool Framegen::load_module(const String &p_path) {
	try {
		device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
		inference_dtype = device.is_cuda() ? torch::kHalf : torch::kFloat32;
		module = torch::jit::load(p_path.utf8().get_data());
		module.to(device);
		if (inference_dtype == torch::kHalf) {
			module.to(torch::kHalf);
		}
		module.eval();
		_ensure_static_buffers();
		module_loaded = true;
		_start_worker();
		return true;
	} catch (const c10::Error &e) {
		print_error("Error loading module: " + String(e.what()));
		return false;
	}
}

torch::Tensor Framegen::_process_player_inputs(Dictionary inp_t) {
	if (!module_loaded) {
		print_error("Module not loaded.");
		return torch::Tensor();
	}

	_ensure_static_buffers();

	Dictionary camera_dict = inp_t.get("camera", Dictionary());
	Array forward_array = camera_dict.get("forward", Array());
	Array up_array = camera_dict.get("up", Array());
	Array rot_array = camera_dict.get("rotation", Array());

	float cam_forward_x = forward_array.size() > 0 ? (float)forward_array[0] : 0.0f;
	float cam_forward_y = forward_array.size() > 1 ? (float)forward_array[1] : 0.0f;
	float cam_forward_z = forward_array.size() > 2 ? (float)forward_array[2] : 0.0f;

	float cam_up_x = up_array.size() > 0 ? (float)up_array[0] : 0.0f;
	float cam_up_y = up_array.size() > 1 ? (float)up_array[1] : 0.0f;
	float cam_up_z = up_array.size() > 2 ? (float)up_array[2] : 0.0f;

	float cam_rot_x = rot_array.size() > 0 ? (float)rot_array[0] : 0.0f;
	float cam_rot_y = rot_array.size() > 1 ? (float)rot_array[1] : 0.0f;
	float cam_rot_z = rot_array.size() > 2 ? (float)rot_array[2] : 0.0f;

	Dictionary physics_dict = inp_t.get("physics", Dictionary());
	Array velocity_array = physics_dict.get("char_velocity", Array());
	float vel_x = velocity_array.size() > 0 ? (float)velocity_array[0] : 0.0f;
	float vel_y = velocity_array.size() > 1 ? (float)velocity_array[1] : 0.0f;
	float vel_z = velocity_array.size() > 2 ? (float)velocity_array[2] : 0.0f;

	Dictionary input_dict = inp_t.get("input", Dictionary());
	float w = (float)input_dict.get("w", 0);
	float a = (float)input_dict.get("a", 0);
	float s = (float)input_dict.get("s", 0);
	float d = (float)input_dict.get("d", 0);
	float m1 = (float)input_dict.get("m1", 0);
	float space = (float)input_dict.get("space", 0);

	float *player_ptr = player_input_staging.data_ptr<float>();
	player_ptr[0] = a;
	player_ptr[1] = d;
	player_ptr[2] = m1;
	player_ptr[3] = s;
	player_ptr[4] = space;
	player_ptr[5] = w;
	player_ptr[6] = vel_x;
	player_ptr[7] = vel_y;
	player_ptr[8] = vel_z;
	player_ptr[9] = cam_rot_x;
	player_ptr[10] = cam_rot_y;
	player_ptr[11] = cam_rot_z;
	player_ptr[12] = cam_forward_x;
	player_ptr[13] = cam_forward_y;
	player_ptr[14] = cam_forward_z;
	player_ptr[15] = cam_up_x;
	player_ptr[16] = cam_up_y;
	player_ptr[17] = cam_up_z;

	player_input_tensor.copy_(player_input_staging.to(device, inference_dtype));

	try {
		return player_input_tensor;
	} catch (const c10::Error &e) {
		print_error("Error processing player inputs: " + String(e.what()));
		return torch::Tensor();
	}
}

torch::Tensor Framegen::_process_input_frame(const Ref<Image> &f_t) {
	if (!module_loaded) {
		print_error("Module not loaded.");
		return torch::Tensor();
	}

	_ensure_static_buffers();

	if (f_t.is_null()) {
		print_error("Input frame is null.");
		return torch::Tensor();
	}

	Ref<Image> img = f_t->duplicate();

	const int expected_width = INPUT_WIDTH;
	const int expected_height = INPUT_HEIGHT;
	if (img->get_width() != expected_width || img->get_height() != expected_height) {
		img->resize(expected_width, expected_height, Image::INTERPOLATE_BILINEAR);
	}

	img->convert(Image::FORMAT_RGB8); // force 3 channels to match Python reshape(..., 3)

	PackedByteArray bytes = img->get_data(); // raw uint8 buffer, like np.fromfile(..., uint8)
	const int64_t expected_size = (int64_t)INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
	if (bytes.size() != expected_size) {
		print_error("Unexpected input frame byte size.");
		return torch::Tensor();
	}

	memcpy(input_staging_u8.data_ptr<uint8_t>(), bytes.ptr(), (size_t)expected_size);

	torch::Tensor normalized = input_staging_u8.to(torch::kFloat32)
									   .div_(255.0f)
									   .permute({ 2, 0, 1 })
									   .unsqueeze(0)
									   .contiguous();

	input_tensor.copy_(normalized.to(device, inference_dtype));

	try {
		return input_tensor;
	} catch (const c10::Error &e) {
		print_error("Error processing input frame: " + String(e.what()));
		return torch::Tensor();
	}
}

Ref<Image> Framegen::_run_inference(const Ref<Image> &f_t, Dictionary inp_t) {
	if (!module_loaded) {
		print_error("Module not loaded.");
		return Ref<Image>();
	}

	input_tensor = _process_input_frame(f_t);
	player_input_tensor = _process_player_inputs(inp_t);

	//int original_width = f_t->get_width();
	//int original_height = f_t->get_height();

	if (!input_tensor.defined() || !player_input_tensor.defined()) {
		print_error("Error processing inputs.");
		return Ref<Image>();
	}

	try {
		torch::InferenceMode inference_mode;
		forward_inputs.clear();
		forward_inputs.push_back(input_tensor);
		forward_inputs.push_back(player_input_tensor);

		torch::Tensor output = module.forward(forward_inputs).toTensor();
		output = output.detach().to(torch::kCPU);

		if (output.dim() == 4 && output.size(0) == 1) {
			output = output.squeeze(0);
		}

		if (output.dim() != 3) {
			print_error("Output tensor must be 3D (CHW or HWC).");
			return Ref<Image>();
		}

		if (output.size(0) == 3) {
			output = output.permute({ 1, 2, 0 });
		} else if (output.size(2) != 3) {
			print_error("Output tensor must have 3 channels (RGB).");
			return Ref<Image>();
		}

		output = output.contiguous().mul(255).clamp(0, 255).to(torch::kUInt8);

		int height = output.size(0);
		int width = output.size(1);
		int channels = output.size(2);

		if (channels != 3) {
			print_error("Output tensor must have 3 channels (RGB).");
			return Ref<Image>();
		}

		torch::Tensor output_u8 = output.contiguous(); // ensure dense CPU layout
		int64_t byte_count = output_u8.numel(); // H * W * 3 for RGB8

		output_buffer.resize(byte_count);
		memcpy(output_buffer.ptrw(), output_u8.data_ptr<uint8_t>(), (size_t)byte_count);

		Ref<Image> img = Image::create_from_data(width, height, false, Image::FORMAT_RGB8, output_buffer);

		/*if (img->get_width() != original_width || img->get_height() != original_height) {
			img->resize(original_width, original_height, Image::INTERPOLATE_BILINEAR);
		}*/

		_publish_present_frame(img);
		return img;
	} catch (const c10::Error &e) {
		print_error("Error generating frame: " + String(e.what()));
		return Ref<Image>();
	} catch (const std::exception &e) {
		print_error("Error generating frame: " + String(e.what()));
		return Ref<Image>();
	}
}

Ref<Image> Framegen::generate_frame(const Ref<Image> &f_t, Dictionary inp_t) {
	std::lock_guard<std::mutex> lock(inference_mutex);
	return _run_inference(f_t, inp_t);
}

bool Framegen::submit_frame(const Ref<Image> &f_t, Dictionary inp_t) {
	if (!module_loaded) {
		print_error("Module not loaded.");
		return false;
	}

	if (f_t.is_null()) {
		return false;
	}

	Ref<Image> frame_copy = f_t->duplicate();
	if (frame_copy.is_null()) {
		return false;
	}

	{
		std::lock_guard<std::mutex> lock(worker_mutex);
		if (!worker_running) {
			return false;
		}
		worker_pending_frame = frame_copy;
		worker_pending_input = inp_t;
		worker_has_job = true;
	}

	worker_cv.notify_one();
	return true;
}

bool Framegen::is_generated_frame_ready() const {
	std::lock_guard<std::mutex> lock(worker_mutex);
	return worker_has_ready_frame;
}

Ref<Image> Framegen::consume_generated_frame() {
	std::lock_guard<std::mutex> lock(worker_mutex);
	if (!worker_has_ready_frame) {
		return Ref<Image>();
	}

	Ref<Image> result = worker_ready_frame;
	worker_ready_frame.unref();
	worker_has_ready_frame = false;
	return result;
}