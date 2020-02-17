global:
    # list of modules to load when starting
    startup: ['man', 'tray', 'tasklogic']

    module_server:
        address: 'localhost'
        port: 12345

    ## For controlling the appearance of the GUI:
    stylesheet: 'qdark.qss'

hardware:
    laserdummy:
        module.Class: 'laser.simple_laser_dummy.SimpleLaserDummy'

logic:
    laserlogic:
        module.Class: 'laser_logic.LaserLogic'
        connect:
            laser: 'laserdummy'

gui:
    laser:
        module.Class: 'laser.laser.LaserGUI'
        connect:
            laserlogic: 'laserlogic'