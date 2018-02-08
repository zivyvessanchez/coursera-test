import win32gui
import re

class Window:
    """Encapsulates some calls to the winapi for window management"""    
    def __init__ (self, wildcard=''):
        """Constructor"""
        self._handle = None
        if wildcard != '':
            self.findWindow(wildcard)
    
    def _windowEnumCallback(self, hwnd, wildcard):
        '''Pass to win32gui.EnumWindows() to check all the opened windows'''
        if re.match(wildcard, str(win32gui.GetWindowText(hwnd))) != None:
            self._handle = hwnd

    def findWindow(self, wildcard):
        self._handle = None
        win32gui.EnumWindows(self._windowEnumCallback, wildcard)

    def getWindowRect(self):
        return win32gui.GetWindowPlacement(self._handle)[-1]
