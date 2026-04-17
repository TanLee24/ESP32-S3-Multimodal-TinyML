#ifndef __VOICE_EI_CONSTANTS__H__
#define __VOICE_EI_CONSTANTS__H__

#define VOICE_EI_CLASSIFIER_RESIZE_NONE                0
#define VOICE_EI_CLASSIFIER_RESIZE_FIT_SHORTEST        1
#define VOICE_EI_CLASSIFIER_RESIZE_FIT_LONGEST         2
#define VOICE_EI_CLASSIFIER_RESIZE_SQUASH              3

// This exists for linux runner, etc
__attribute__((unused)) static const char *VOICE_EI_RESIZE_STRINGS[] = { "none", "fit-shortest", "fit-longest", "squash" };

#endif  //!__VOICE_EI_CONSTANTS__H__