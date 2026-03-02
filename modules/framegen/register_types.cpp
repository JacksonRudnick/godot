#include "register_types.h"

#include "core/object/class_db.h"
#include "framegen.h"

void initialize_framegen_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<Framegen>();
}

void uninitialize_framegen_module(ModuleInitializationLevel p_level) {
}