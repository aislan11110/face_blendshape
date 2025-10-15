# Copyright: (C) Robotics Brain and Cognitive Sciences (RBCS) - Istituto Italiano di Tecnologia
# CopyPolicy: Released under the terms of the GNU GPL v2.0.

set(MODULE_NAME faceBlendshapes)

message("Installing ${MODULE_NAME}")

# Path to the main Python script
set(PATH_TO_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/main.py)

# Set the name of the script file to be generated
set(SCRIPT_FILE_NAME ${MODULE_NAME})

# Set the command to activate the virtual environment and run the Python script
set(SCRIPT_CONTENT "#!/bin/bash\n\n# Function to terminate the child process when the bash father is terminated \nterminate_child_process() {\n    kill $PID\n}\n\n# Catch the quitting signal and send it to the child process\ntrap terminate_child_process EXIT\n\n# Run python script\n${CONDA_ENV_PATH}/bin/python3 ${PATH_TO_SCRIPT} \"$@\"  &\nPID=$!\n\n# Wait for the python process to end\nwait $PID")

# Generate the script file
file(WRITE ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME} ${SCRIPT_CONTENT})

# Make the script file executable
execute_process(COMMAND chmod +x ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME})

message("Your ${SCRIPT_FILE_NAME} script has been created")

find_package(ICUBcontrib REQUIRED)
include(ICUBcontribHelpers)
include(ICUBcontribOptions)
icubcontrib_set_default_prefix()

SET(CMAKE_MODULE_PATH ${ICUB_MODULE_PATH} ${CMAKE_MODULE_PATH})

message(STATUS "Installing ${MODULE_NAME} in ${ICUBcontrib_DIR}/bin/${SCRIPT_FILE_NAME}")

find_package(YARP REQUIRED)
FIND_PACKAGE(ICUBcontrib REQUIRED)
list(APPEND CMAKE_MODULE_PATH ${ICUBCONTRIB_MODULE_PATH})
include(ICUBcontribHelpers)
include(ICUBcontribOptions)

icubcontrib_set_default_prefix()
set(ICUB_APPLICATIONS_PREFIX "$ENV{ICUB_ROOT}" CACHE PATH "Application path prefix")

set(PROGRAM_PERMISSIONS_DEFAULT
    OWNER_WRITE OWNER_READ OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

install(FILES ${CMAKE_BINARY_DIR}/bin/${SCRIPT_FILE_NAME}
        PERMISSIONS ${PROGRAM_PERMISSIONS_DEFAULT}
        DESTINATION bin)

# Install the MediaPipe model file
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/face_landmarker.task
        DESTINATION share/${MODULE_NAME})