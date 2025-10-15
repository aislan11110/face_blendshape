# Copyright: (C) 2023 Istituto Italiano di Tecnologia | Robotics, Brains and Cognitive Science
# Authors: Luca Garello
# CopyPolicy: Released under the terms of the GNU GPL v2.0.


# NOTE:
# The .sh files and the .ini files are installed in context folders named after the appname. 
# The .xml files and the .xml templates are installed in the icubcontrib applications folder.

set(appname ${PROJECT_NAME})

file(GLOB demos ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.sh)
file(GLOB conf ${CMAKE_CURRENT_SOURCE_DIR}/conf/*.ini)
file(GLOB app ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.xml)
file(GLOB app_template ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.template)

yarp_install(PROGRAMS ${demos} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
yarp_install(FILES ${conf} DESTINATION ${ICUBCONTRIB_CONTEXTS_INSTALL_DIR}/${appname})
yarp_install(FILES ${app} DESTINATION ${ICUBCONTRIB_APPLICATIONS_INSTALL_DIR})
yarp_install(FILES ${app_template} DESTINATION ${ICUBCONTRIB_APPLICATIONS_TEMPLATES_INSTALL_DIR})

