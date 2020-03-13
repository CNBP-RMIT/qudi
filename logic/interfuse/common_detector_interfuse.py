"""
This file contains the Qudi Interfuse to control the scanner and counting logic with the same detector.

Qudi is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Qudi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Qudi. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Qudi Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution and at <https://github.com/Ulm-IQO/qudi/>
"""

from core.connector import Connector
from logic.generic_logic import GenericLogic
from interface.slow_counter_interface import SlowCounterInterface
from interface.confocal_scanner_interface import ConfocalScannerInterface


class CommonDetectorInterfuse(GenericLogic, SlowCounterInterface, ConfocalScannerInterface):
    """ This interfuse allows the simultaneous control of the scanner and counter logic.
    The counter will be automatically interrupted when a scanner is started and will resume when the scan ends.
    """

    confocalscanner = Connector(interface='ConfocalScannerInterface')
    counter = Connector(interface='SlowCounterInterface')

    internal_state = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._scanning_device = self.confocalscanner()
        self._counting_device = self.counter()

        # The internal state is used as an extension to the default state machine status
        # the 'idle' and 'locked' states works like in other modules
        # the 'interrutable' and 'interrupted' states are used to make the counter and the scanner interact
        self.internal_state = 'idle'

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        pass

    # =================== SlowCounterInterface Commands ========================
    def get_constraints(self):
        """ Get hardware limits of NI device.

        @return SlowCounterConstraints: constraints class for slow counter
        """
        pass

    def set_up_clock(self, clock_frequency=None, clock_channel=None, scanner=False, idle=False):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of
                                      the clock in Hz
        @param string clock_channel: if defined, this is the physical channel
                                     of the clock within the NI card.
        @param bool scanner: if set to True method will set up a clock function
                             for the scanner, otherwise a clock function for a
                             counter will be set.
        @param bool idle: set whether idle situation of the counter (where
                          counter is doing nothing) is defined as
                                True  = 'Voltage High/Rising Edge'
                                False = 'Voltage Low/Falling Edge'

        @return int: error code (0:OK, -1:error)
        """
        pass

    def set_up_counter(self,
                       counter_channels=None,
                       sources=None,
                       clock_channel=None,
                       counter_buffer=None):
        """ Configures the actual counter with a given clock.

        @param list(str) counter_channels: optional, physical channel of the counter
        @param list(str) sources: optional, physical channel where the photons
                                  are to count from
        @param str clock_channel: optional, specifies the clock channel for the
                                  counter
        @param int counter_buffer: optional, a buffer of specified integer
                                   length, where in each bin the count numbers
                                   are saved.

        @return int: error code (0:OK, -1:error)
        """
        pass

    def get_counter(self, samples=None):
        """ Returns the current counts per second of the counter.

        @param int samples: if defined, number of samples to read in one go.
                            How many samples are read per readout cycle. The
                            readout frequency was defined in the counter setup.
                            That sets also the length of the readout array.

        @return float [samples]: array with entries as photon counts per second
        """
        pass

    def get_counter_channels(self):
        """ Returns the list of counter channel names.

        @return tuple(str): channel names

        Most methods calling this might just care about the number of channels, though.
        """
        pass

    def close_counter(self, scanner=False):
        """ Closes the counter or scanner and cleans up afterwards.

        @param bool scanner: specifies if the counter- or scanner- function
                             will be excecuted to close the device.
                                True = scanner
                                False = counter

        @return int: error code (0:OK, -1:error)
        """
        pass

    def close_clock(self, scanner=False):
        """ Closes the clock and cleans up afterwards.

        @param bool scanner: specifies if the counter- or scanner- function
                             should be used to close the device.
                                True = scanner
                                False = counter

        @return int: error code (0:OK, -1:error)
        """
        pass
    # ================ End SlowCounterInterface Commands =======================

    # ================ ConfocalScannerInterface Commands =======================
    def reset_hardware(self):
        """ Resets the hardware, so the connection is lost and other programs
            can access it.

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.reset_hardware()

    def get_position_range(self):
        """ Returns the physical range of the scanner.

        @return float [4][2]: array of 4 ranges with an array containing lower
                              and upper limit
        """
        return self._scanning_device.get_position_range()

    def set_position_range(self, myrange=None):
        """ Sets the physical range of the scanner.

        @param float [4][2] myrange: array of 4 ranges with an array containing
                                     lower and upper limit

        @return int: error code (0:OK, -1:error)
        """
        if myrange is None:
            myrange = [[0, 1], [0, 1], [0, 1], [0, 1]]
        return self._scanning_device.set_position_range(myrange)

    def set_voltage_range(self, myrange=None):
        """ Sets the voltage range of the NI Card.

        @param float [2] myrange: array containing lower and upper limit

        @return int: error code (0:OK, -1:error)
        """
        if myrange is None:
            myrange = [-10., 10.]
        return self._scanning_device.set_voltage_range(myrange)

    def get_scanner_axes(self):
        """ Pass through scanner axes """
        return self._scanning_device.get_scanner_axes()

    def get_scanner_count_channels(self):
        """ Pass through scanner counting channels """
        return self._scanning_device.get_scanner_count_channels()

    def set_up_scanner_clock(self, clock_frequency=None, clock_channel=None):
        """ Configures the hardware clock of the NiDAQ card to give the timing.

        @param float clock_frequency: if defined, this sets the frequency of the
                                      clock
        @param str clock_channel: if defined, this is the physical channel of
                                  the clock

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.set_up_scanner_clock(clock_frequency, clock_channel)

    def set_up_scanner(self, counter_channel=None, photon_source=None,
                       clock_channel=None, scanner_ao_channels=None):
        """ Configures the actual scanner with a given clock.

        @param str counter_channel: if defined, this is the physical channel
                                    of the counter
        @param str photon_source: if defined, this is the physical channel where
                                  the photons are to count from
        @param str clock_channel: if defined, this specifies the clock for the
                                  counter
        @param str scanner_ao_channels: if defined, this specifies the analoque
                                        output channels

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.set_up_scanner(
            counter_channel,
            photon_source,
            clock_channel,
            scanner_ao_channels)

    def scanner_set_position(self, x=None, y=None, z=None, a=None):
        """Move stage to x, y, z, a (where a is the fourth voltage channel).

        @param float x: position in x-direction (volts)
        @param float y: position in y-direction (volts)
        @param float z: position in z-direction (volts)
        @param float a: position in a-direction (volts)

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.scanenr_set_position(x, y, z, a)

    def get_scanner_position(self):
        """ Get the current position of the scanner hardware.

        @return float[]: current position in (x, y, z, a).
        """
        return self._scanning_device.get_scanner_position()

    def set_up_line(self, length=100):
        """ Sets up the analog output for scanning a line.

        @param int length: length of the line in pixel

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.set_up_line(length)

    def scan_line(self, line_path=None, pixel_clock=False):
        """ Scans a line and returns the counts on that line.

        @param float[][4] line_path: array of 4-part tuples defining the positions pixels
        @param bool pixel_clock: whether we need to output a pixel clock for this line

        @return float[]: the photon counts per second
        """
        return self._scanning_device.scan_line(line_path, pixel_clock)

    def close_scanner(self):
        """ Closes the scanner and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.close_scanner()

    def close_scanner_clock(self, power=0):
        """ Closes the clock and cleans up afterwards.

        @return int: error code (0:OK, -1:error)
        """
        return self._scanning_device.close_scanner_clock()
    # ================ End ConfocalScannerInterface Commands ===================

    def isInterruptable(self):
        """ Check if the detector is performing an interrputable action (i.e. counting)
        """
        return self.internal_state == 'interruptable'

    def isInterrupted(self):
        """ Check if the detector was interrupted by another action (i.e. scanning or performing ODMR)
        """
        return self.internal_state == 'interrupted'

    def interrupt(self):
        """ Interrupt the counting if it is running so that another action can start with the same detector
        """
        pass

    def release(self):
        """ Restart using the detector for counting if it was previously interrupted
        """
        pass
