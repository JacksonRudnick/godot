#include "framegen_present_bridge.h"

#include "framegen.h"

bool framegen_consume_latest_present_frame(Ref<Image> &r_img) {
	return Framegen::consume_latest_present_frame(r_img);
}
