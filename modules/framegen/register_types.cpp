#include "register_types.h"

#include "core/object/class_db.h"
#include "framegen.h"
#include "framegen_present_bridge.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"

void initialize_framegen_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<Framegen>();
	renderer_compositor_rd_set_framegen_consume_callback(&framegen_consume_latest_present_frame);
}

void uninitialize_framegen_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	renderer_compositor_rd_set_framegen_consume_callback(nullptr);
}