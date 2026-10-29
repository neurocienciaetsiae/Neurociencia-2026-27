#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.1.3),
    on junio 10, 2026, at 14:13
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_labrecorder
from pylsl import StreamInfo, StreamOutlet
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.1.3'
expName = 'Trust ATC'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1910, 1070]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\user\\Desktop\\Trust ATC definitive\\Trust ATC_neutral_100_100_50_100_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('debug')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "lab_recorder" ---
    text_labrecorder = visual.TextStim(win=win, name='text_labrecorder',
        text='Please connect the LSL stream to Lab Recorder',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_labrecorder = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_labrecorder = visual.TextStim(win=win, name='text_space_labrecorder',
        text='Press SPACE when connected',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from code_labrecorder
    try:
        from pylsl import StreamInfo, StreamOutlet
    
        info = StreamInfo(
            name='Markers',
            type='Markers',
            channel_count=1,
            nominal_srate=0,
            channel_format='string',
            source_id='trust_atc_markers_001'
        )
    
        outlet = StreamOutlet(info)
        outlet.push_sample(['EXPERIMENT_INIT'])
    
        lsl_ok = True
    
    except Exception as e:
        print(f"[LSL] Could not initialize pylsl: {e}")
        outlet = None
        lsl_ok = False
    
    # --- Initialize components for Routine "welcome" ---
    Welcome = visual.TextStim(win=win, name='Welcome',
        text='Welcome to the experiment on trust in ATC automation',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_welcome = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_welcome = visual.TextStim(win=win, name='text_space_welcome',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "calibration_instructions_closed" ---
    text_calibration_instructions_closed = visual.TextStim(win=win, name='text_calibration_instructions_closed',
        text='The device calibration will now begin. Please keep your eyes closed for one minute. A beep will indicate when you may open them.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_calibration_instructions_closed = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_calibration_instructions_closed = visual.TextStim(win=win, name='text_space_calibration_instructions_closed',
        text='Press SPACE and close your eyes to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "calibration_time_closed" ---
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "calibration_instructions_open" ---
    text_calibration_instructions_open = visual.TextStim(win=win, name='text_calibration_instructions_open',
        text='Please keep your eyes open for one minute. A beep will indicate the end of the calibration.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_calibration_instructions_open = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_calibration_instructions_open = visual.TextStim(win=win, name='text_space_calibration_instructions_open',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # set audio backend
    sound.Sound.backend = 'ptb'
    sound_closed = sound.Sound(
        'A', 
        secs=0.5, 
        stereo=True, 
        hamming=True, 
        speaker=None,    name='sound_closed'
    )
    sound_closed.setVolume(1.0)
    
    # --- Initialize components for Routine "calibration_time_open" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instructions_willingness" ---
    text_instructions_willingness = visual.TextStim(win=win, name='text_instructions_willingness',
        text='Next, you will complete a questionnaire assessing willingness to use automation in the air traffic control environment. \nThe questionnaire consists of 6 questions.\n Please select your answers on a scale from 1 (strongly disagree) to 5 (strongly agree).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instructions_willingness = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_instructions_willingness = visual.TextStim(win=win, name='text_space_instructions_willingness',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_willingness" ---
    text_willingness = visual.TextStim(win=win, name='text_willingness',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_willingness = visual.Slider(win=win, name='slider_willingness',
        startValue=None, size=(1.0, 0.1), pos=(0, -0.1), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    mouse_willingness = event.Mouse(win=win)
    x, y = [None, None]
    mouse_willingness.mouseClock = core.Clock()
    prog_willingness = visual.Progress(
        win, name='prog_willingness',
        progress=0.0,
        pos=(-0.5, -0.3), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-4
    )
    
    # --- Initialize components for Routine "instructions_trial" ---
    text_instructions_trial = visual.TextStim(win=win, name='text_instructions_trial',
        text='Thank you for completing the initial questionnaires.\nNext, you will be required to complete a task related to vertical separation.\nA fixation cross will indicate when you should look at the screen to complete it.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instructions_trial = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_instructions_trial = visual.TextStim(win=win, name='text_space_instructions_trial',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instructions_trial_2" ---
    key_resp_instructions_trial_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_instructions_trial_2 = visual.TextStim(win=win, name='text_space_instructions_trial_2',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    text_instructions_trial_2 = visual.TextStim(win=win, name='text_instructions_trial_2',
        text='The task consists of preventing two aircraft from violating the minimum vertical separation standards (<1000 ft, i.e., between 1 and 9 FL).\nThe flight level of each aircraft will be displayed on the screen, and you must decide whether to change the flight level or maintain it.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instructions_trial_3" ---
    text_instrucciones_ejercicio__3 = visual.TextStim(win=win, name='text_instrucciones_ejercicio__3',
        text='The experiment is divided into three blocks: a manual task and four blocks including an automated decision-support aid.\nThe system will display a recommendation indicating whether the flight level should be changed or maintained.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instrucciones_ejercicio_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_instrucciones_ejercicio_3 = visual.TextStim(win=win, name='text_space_instrucciones_ejercicio_3',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instructions_trial_4" ---
    text_instructions_trial_4 = visual.TextStim(win=win, name='text_instructions_trial_4',
        text='In addition, you will be asked to indicate how much you trust the automation after each response and complete a brief questionnaire after each block involving the automated aid.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_instructions_trial_4 = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_instructions_trial_4 = visual.TextStim(win=win, name='text_space_instructions_trial_4',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "trial_manual_start" ---
    text_trial_manual_start = visual.TextStim(win=win, name='text_trial_manual_start',
        text='MANUAL TASK\n\nRemember that the required separation is 10 FL or more. You have 3 seconds to make your decision.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_trial_manual_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_trial_manual_start = visual.TextStim(win=win, name='text_space_trial_manual_start',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "time_manual" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_look = visual.ShapeStim(
        win=win, name='fixation_cross_look', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(-1.0000, -1.0000, -1.0000), fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_manual" ---
    trial_manual_aircraft_1 = visual.TextStim(win=win, name='trial_manual_aircraft_1',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_manual_aircraft_2 = visual.TextStim(win=win, name='trial_manual_aircraft_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    change_manual = visual.ButtonStim(win, 
        text='Change FL', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#bb0000', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='change_manual',
        depth=-2
    )
    change_manual.buttonClock = core.Clock()
    maintain_manual = visual.ButtonStim(win, 
        text='Maintain FL', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#004bbb', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='maintain_manual',
        depth=-3
    )
    maintain_manual.buttonClock = core.Clock()
    
    # --- Initialize components for Routine "answer_manual" ---
    text_answer_manual = visual.TextStim(win=win, name='text_answer_manual',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color=[-0.7700, -0.7700, -0.7500], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "high_reliability_start" ---
    text_high_reliability_start = visual.TextStim(win=win, name='text_high_reliability_start',
        text='AUTOMATED DECISION SUPPORT\n\nThe task is the same, but the automated aid will be displayed as a bar located below the recommended option.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_high_reliability_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_high_reliability_start = visual.TextStim(win=win, name='text_space_high_reliability_start',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "time_high" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_look = visual.ShapeStim(
        win=win, name='fixation_cross_look', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(-1.0000, -1.0000, -1.0000), fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_high" ---
    trial_high_aircraft_1 = visual.TextStim(win=win, name='trial_high_aircraft_1',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_high_aircraft_2 = visual.TextStim(win=win, name='trial_high_aircraft_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    change_high = visual.ButtonStim(win, 
        text='Change FL', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#bb0000', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='change_high',
        depth=-2
    )
    change_high.buttonClock = core.Clock()
    maintain_high = visual.ButtonStim(win, 
        text='Maintain FL', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#004bbb', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='maintain_high',
        depth=-3
    )
    maintain_high.buttonClock = core.Clock()
    support_high = visual.Rect(
        win=win, name='support_high',
        width=(0.4, 0.03)[0], height=(0.4, 0.03)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(1.0000, 1.0000, 1.0000), fillColor='#262626',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "answer_high" ---
    text_answer_high = visual.TextStim(win=win, name='text_answer_high',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_high" ---
    text_high_question = visual.TextStim(win=win, name='text_high_question',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_high_question = visual.Slider(win=win, name='slider_high_question',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[0.6549, 0.6549, 0.6549], markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    mouse_high_question = event.Mouse(win=win)
    x, y = [None, None]
    mouse_high_question.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "SATI_short_high_instructions" ---
    text_SATI_short_high_instructions = visual.TextStim(win=win, name='text_SATI_short_high_instructions',
        text='Next, you will complete a questionnaire on trust in the automated aid consisting of 6 questions.\nPlease select your answers on a scale from 1 (strongly disagree) to 5 (strongly agree).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_SATI_short_high_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_SATI_short_high_instructions = visual.TextStim(win=win, name='text_space_SATI_short_high_instructions',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "SATI_short_high" ---
    text_SATI_short_high = visual.TextStim(win=win, name='text_SATI_short_high',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_SATI_short_high_clarification = visual.TextStim(win=win, name='text_SATI_short_high_clarification',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_SATI_short_high = visual.Slider(win=win, name='slider_SATI_short_high',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    mouse_SATI_short_high = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SATI_short_high.mouseClock = core.Clock()
    prog_short_high = visual.Progress(
        win, name='prog_short_high',
        progress=0.0,
        pos=(-0.5, -0.25), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "how_you_feel_high" ---
    pleasant_high = visual.ButtonStim(win, 
        text='Pleasant', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='pleasant_high',
        depth=0
    )
    pleasant_high.buttonClock = core.Clock()
    unpleasant_high = visual.ButtonStim(win, 
        text='Unpleasant', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='unpleasant_high',
        depth=-1
    )
    unpleasant_high.buttonClock = core.Clock()
    text_how_you_feel_high = visual.TextStim(win=win, name='text_how_you_feel_high',
        text='How did you find the task?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pleasant_high_choice" ---
    excited_high = visual.ButtonStim(win, 
        text='Excited', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='excited_high',
        depth=0
    )
    excited_high.buttonClock = core.Clock()
    relaxed_high = visual.ButtonStim(win, 
        text='Relaxed', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='relaxed_high',
        depth=-1
    )
    relaxed_high.buttonClock = core.Clock()
    text_pleasant_high = visual.TextStim(win=win, name='text_pleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "unpleasant_high_choice" ---
    stressed_high = visual.ButtonStim(win, 
        text='Stressed', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='stressed_high',
        depth=0
    )
    stressed_high.buttonClock = core.Clock()
    bored_high = visual.ButtonStim(win, 
        text='Bored', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='bored_high',
        depth=-1
    )
    bored_high.buttonClock = core.Clock()
    text_unpleasant_high = visual.TextStim(win=win, name='text_unpleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "high_reliability_start" ---
    text_high_reliability_start = visual.TextStim(win=win, name='text_high_reliability_start',
        text='AUTOMATED DECISION SUPPORT\n\nThe task is the same, but the automated aid will be displayed as a bar located below the recommended option.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_high_reliability_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_high_reliability_start = visual.TextStim(win=win, name='text_space_high_reliability_start',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "time_high" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_look = visual.ShapeStim(
        win=win, name='fixation_cross_look', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(-1.0000, -1.0000, -1.0000), fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_high" ---
    trial_high_aircraft_1 = visual.TextStim(win=win, name='trial_high_aircraft_1',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_high_aircraft_2 = visual.TextStim(win=win, name='trial_high_aircraft_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    change_high = visual.ButtonStim(win, 
        text='Change FL', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#bb0000', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='change_high',
        depth=-2
    )
    change_high.buttonClock = core.Clock()
    maintain_high = visual.ButtonStim(win, 
        text='Maintain FL', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#004bbb', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='maintain_high',
        depth=-3
    )
    maintain_high.buttonClock = core.Clock()
    support_high = visual.Rect(
        win=win, name='support_high',
        width=(0.4, 0.03)[0], height=(0.4, 0.03)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(1.0000, 1.0000, 1.0000), fillColor='#262626',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "answer_high" ---
    text_answer_high = visual.TextStim(win=win, name='text_answer_high',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_high" ---
    text_high_question = visual.TextStim(win=win, name='text_high_question',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_high_question = visual.Slider(win=win, name='slider_high_question',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[0.6549, 0.6549, 0.6549], markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    mouse_high_question = event.Mouse(win=win)
    x, y = [None, None]
    mouse_high_question.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "SATI_short_high_instructions" ---
    text_SATI_short_high_instructions = visual.TextStim(win=win, name='text_SATI_short_high_instructions',
        text='Next, you will complete a questionnaire on trust in the automated aid consisting of 6 questions.\nPlease select your answers on a scale from 1 (strongly disagree) to 5 (strongly agree).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_SATI_short_high_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_SATI_short_high_instructions = visual.TextStim(win=win, name='text_space_SATI_short_high_instructions',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "SATI_short_high" ---
    text_SATI_short_high = visual.TextStim(win=win, name='text_SATI_short_high',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_SATI_short_high_clarification = visual.TextStim(win=win, name='text_SATI_short_high_clarification',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_SATI_short_high = visual.Slider(win=win, name='slider_SATI_short_high',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    mouse_SATI_short_high = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SATI_short_high.mouseClock = core.Clock()
    prog_short_high = visual.Progress(
        win, name='prog_short_high',
        progress=0.0,
        pos=(-0.5, -0.25), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "how_you_feel_high" ---
    pleasant_high = visual.ButtonStim(win, 
        text='Pleasant', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='pleasant_high',
        depth=0
    )
    pleasant_high.buttonClock = core.Clock()
    unpleasant_high = visual.ButtonStim(win, 
        text='Unpleasant', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='unpleasant_high',
        depth=-1
    )
    unpleasant_high.buttonClock = core.Clock()
    text_how_you_feel_high = visual.TextStim(win=win, name='text_how_you_feel_high',
        text='How did you find the task?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pleasant_high_choice" ---
    excited_high = visual.ButtonStim(win, 
        text='Excited', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='excited_high',
        depth=0
    )
    excited_high.buttonClock = core.Clock()
    relaxed_high = visual.ButtonStim(win, 
        text='Relaxed', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='relaxed_high',
        depth=-1
    )
    relaxed_high.buttonClock = core.Clock()
    text_pleasant_high = visual.TextStim(win=win, name='text_pleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "unpleasant_high_choice" ---
    stressed_high = visual.ButtonStim(win, 
        text='Stressed', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='stressed_high',
        depth=0
    )
    stressed_high.buttonClock = core.Clock()
    bored_high = visual.ButtonStim(win, 
        text='Bored', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='bored_high',
        depth=-1
    )
    bored_high.buttonClock = core.Clock()
    text_unpleasant_high = visual.TextStim(win=win, name='text_unpleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "low_reliability_start" ---
    text_low_reliability_start = visual.TextStim(win=win, name='text_low_reliability_start',
        text='AUTOMATED DECISION SUPPORT\n\nThe task is the same, but the automated aid will be displayed as a bar located below the recommended option.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_low_reliability_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_low_reliability_start = visual.TextStim(win=win, name='text_space_low_reliability_start',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "time_low" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_look = visual.ShapeStim(
        win=win, name='fixation_cross_look', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(-1.0000, -1.0000, -1.0000), fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_low" ---
    trial_low_aircraft_1 = visual.TextStim(win=win, name='trial_low_aircraft_1',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_low_aircraft_2 = visual.TextStim(win=win, name='trial_low_aircraft_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    change_low = visual.ButtonStim(win, 
        text='Change FL', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#bb0000', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='change_low',
        depth=-2
    )
    change_low.buttonClock = core.Clock()
    maintain_low = visual.ButtonStim(win, 
        text='Maintain FL', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#004bbb', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='maintain_low',
        depth=-3
    )
    maintain_low.buttonClock = core.Clock()
    support_low = visual.Rect(
        win=win, name='support_low',
        width=(0.4, 0.03)[0], height=(0.4, 0.03)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[1.0000, 1.0000, 1.0000], fillColor='#262626',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "answer_low" ---
    text_answer_low = visual.TextStim(win=win, name='text_answer_low',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color=[-0.7700, -0.7700, -0.7500], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_low" ---
    text_low_question = visual.TextStim(win=win, name='text_low_question',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_low_question = visual.Slider(win=win, name='slider_low_question',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    mouse_low_question = event.Mouse(win=win)
    x, y = [None, None]
    mouse_low_question.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "SATI_short_low_instructions" ---
    text_SATI_short_low_instructions = visual.TextStim(win=win, name='text_SATI_short_low_instructions',
        text='Next, you will complete a questionnaire assessing trust in the automated aid. The questionnaire consists of 6 questions.\nPlease select your answers on a scale from 1 (strongly disagree) to 5 (strongly agree).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_SATI_short_low_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_SATI_short_low_instructions = visual.TextStim(win=win, name='text_space_SATI_short_low_instructions',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "SATI_short_low" ---
    text_SATI_short_low = visual.TextStim(win=win, name='text_SATI_short_low',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_SATI_short_low_clarification = visual.TextStim(win=win, name='text_SATI_short_low_clarification',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_SATI_short_low = visual.Slider(win=win, name='slider_SATI_short_low',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    mouse_SATI_short_low = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SATI_short_low.mouseClock = core.Clock()
    prog_short_low = visual.Progress(
        win, name='prog_short_low',
        progress=0.0,
        pos=(-0.5, -0.25), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "how_you_feel_low" ---
    pleasant_low = visual.ButtonStim(win, 
        text='Pleasant', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='pleasant_low',
        depth=0
    )
    pleasant_low.buttonClock = core.Clock()
    unpleasant_low = visual.ButtonStim(win, 
        text='Unpleasant', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='unpleasant_low',
        depth=-1
    )
    unpleasant_low.buttonClock = core.Clock()
    text_how_you_feel_low = visual.TextStim(win=win, name='text_how_you_feel_low',
        text='How did you find the exercise?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pleasant_low_choice" ---
    excited_low = visual.ButtonStim(win, 
        text='Excited', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='excited_low',
        depth=0
    )
    excited_low.buttonClock = core.Clock()
    relaxed_low = visual.ButtonStim(win, 
        text='Relaxed', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='relaxed_low',
        depth=-1
    )
    relaxed_low.buttonClock = core.Clock()
    text_pleasant_low = visual.TextStim(win=win, name='text_pleasant_low',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "unpleasant_low_choice" ---
    stressed_low = visual.ButtonStim(win, 
        text='Stressed', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='stressed_low',
        depth=0
    )
    stressed_low.buttonClock = core.Clock()
    bored_low = visual.ButtonStim(win, 
        text='Bored', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='bored_low',
        depth=-1
    )
    bored_low.buttonClock = core.Clock()
    text_unpleasant_low = visual.TextStim(win=win, name='text_unpleasant_low',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "high_reliability_start" ---
    text_high_reliability_start = visual.TextStim(win=win, name='text_high_reliability_start',
        text='AUTOMATED DECISION SUPPORT\n\nThe task is the same, but the automated aid will be displayed as a bar located below the recommended option.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_high_reliability_start = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_high_reliability_start = visual.TextStim(win=win, name='text_space_high_reliability_start',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "time_high" ---
    
    # --- Initialize components for Routine "fixation_cross" ---
    fixation_cross_look = visual.ShapeStim(
        win=win, name='fixation_cross_look', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(-1.0000, -1.0000, -1.0000), fillColor=(-1.0000, -1.0000, -1.0000),
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "trial_high" ---
    trial_high_aircraft_1 = visual.TextStim(win=win, name='trial_high_aircraft_1',
        text='',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    trial_high_aircraft_2 = visual.TextStim(win=win, name='trial_high_aircraft_2',
        text='',
        font='Arial',
        pos=(0, 0.1), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    change_high = visual.ButtonStim(win, 
        text='Change FL', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#bb0000', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='change_high',
        depth=-2
    )
    change_high.buttonClock = core.Clock()
    maintain_high = visual.ButtonStim(win, 
        text='Maintain FL', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#004bbb', borderColor=(-1.0000, -1.0000, -1.0000),
        color=[245, 245, 245], colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='maintain_high',
        depth=-3
    )
    maintain_high.buttonClock = core.Clock()
    support_high = visual.Rect(
        win=win, name='support_high',
        width=(0.4, 0.03)[0], height=(0.4, 0.03)[1],
        ori=0.0, pos=[0,0], draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(1.0000, 1.0000, 1.0000), fillColor='#262626',
        opacity=None, depth=-4.0, interpolate=True)
    
    # --- Initialize components for Routine "answer_high" ---
    text_answer_high = visual.TextStim(win=win, name='text_answer_high',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question_high" ---
    text_high_question = visual.TextStim(win=win, name='text_high_question',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    slider_high_question = visual.Slider(win=win, name='slider_high_question',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor=[0.6549, 0.6549, 0.6549], markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    mouse_high_question = event.Mouse(win=win)
    x, y = [None, None]
    mouse_high_question.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "SATI_short_high_instructions" ---
    text_SATI_short_high_instructions = visual.TextStim(win=win, name='text_SATI_short_high_instructions',
        text='Next, you will complete a questionnaire on trust in the automated aid consisting of 6 questions.\nPlease select your answers on a scale from 1 (strongly disagree) to 5 (strongly agree).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_SATI_short_high_instructions = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_SATI_short_high_instructions = visual.TextStim(win=win, name='text_space_SATI_short_high_instructions',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "SATI_short_high" ---
    text_SATI_short_high = visual.TextStim(win=win, name='text_SATI_short_high',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_SATI_short_high_clarification = visual.TextStim(win=win, name='text_SATI_short_high_clarification',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_SATI_short_high = visual.Slider(win=win, name='slider_SATI_short_high',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    mouse_SATI_short_high = event.Mouse(win=win)
    x, y = [None, None]
    mouse_SATI_short_high.mouseClock = core.Clock()
    prog_short_high = visual.Progress(
        win, name='prog_short_high',
        progress=0.0,
        pos=(-0.5, -0.25), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "how_you_feel_high" ---
    pleasant_high = visual.ButtonStim(win, 
        text='Pleasant', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='pleasant_high',
        depth=0
    )
    pleasant_high.buttonClock = core.Clock()
    unpleasant_high = visual.ButtonStim(win, 
        text='Unpleasant', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.45, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='unpleasant_high',
        depth=-1
    )
    unpleasant_high.buttonClock = core.Clock()
    text_how_you_feel_high = visual.TextStim(win=win, name='text_how_you_feel_high',
        text='How did you find the task?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "pleasant_high_choice" ---
    excited_high = visual.ButtonStim(win, 
        text='Excited', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='excited_high',
        depth=0
    )
    excited_high.buttonClock = core.Clock()
    relaxed_high = visual.ButtonStim(win, 
        text='Relaxed', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='relaxed_high',
        depth=-1
    )
    relaxed_high.buttonClock = core.Clock()
    text_pleasant_high = visual.TextStim(win=win, name='text_pleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "unpleasant_high_choice" ---
    stressed_high = visual.ButtonStim(win, 
        text='Stressed', font='Arial',
        pos=(-0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='stressed_high',
        depth=0
    )
    stressed_high.buttonClock = core.Clock()
    bored_high = visual.ButtonStim(win, 
        text='Bored', font='Arial',
        pos=(0.3, -0.2),
        letterHeight=0.05,
        size=(0.4, 0.15), 
        ori=0.0
        ,borderWidth=0.0,
        fillColor='#262626', borderColor=(-1.0000, -1.0000, -1.0000),
        color='white', colorSpace='rgb',
        opacity=None,
        bold=True, italic=False,
        padding=None,
        anchor='center',
        name='bored_high',
        depth=-1
    )
    bored_high.buttonClock = core.Clock()
    text_unpleasant_high = visual.TextStim(win=win, name='text_unpleasant_high',
        text='How do you feel?',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_final_instructions" ---
    text_test_final_instructions = visual.TextStim(win=win, name='text_test_final_instructions',
        text='Finally, you will complete a questionnaire consisting of 8 questions about trust in automation throughout the task.\nPlease select your answer on a scale from 1 (Not at all) to 5 (Absolutely).',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color=[-0.7700, -0.7700, -0.7500], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_test_final_instrucciones = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_test_final_instrucciones = visual.TextStim(win=win, name='text_space_test_final_instrucciones',
        text='Press SPACE to continue',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "test_final" ---
    text_test_final = visual.TextStim(win=win, name='text_test_final',
        text='',
        font='Arial',
        pos=(0, 0.2), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_test_final_clarification = visual.TextStim(win=win, name='text_test_final_clarification',
        text='',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    slider_test_final = visual.Slider(win=win, name='slider_test_final',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(1, 2, 3, 4, 5), ticks=(1, 2, 3, 4, 5), granularity=1.0,
        style='rating', styleTweaks=[], opacity=None,
        labelColor='LightGray', markerColor='#bb0000', lineColor='#262626', colorSpace='rgb',
        font='Noto Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    mouse_test_final = event.Mouse(win=win)
    x, y = [None, None]
    mouse_test_final.mouseClock = core.Clock()
    prog_final = visual.Progress(
        win, name='prog_final',
        progress=0.0,
        pos=(-0.5, -0.25), size=(1, 0.03), anchor='center-left', units='height',
        barColor='#004bbb', backColor=[0.6549, 0.6549, 0.6549], borderColor=[0.6549, 0.6549, 0.6549], colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-5
    )
    
    # --- Initialize components for Routine "End" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='The experiment has ended. Thank you for participating!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.07, wrapWidth=None, ori=0.0, 
        color='#262626', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_end = keyboard.Keyboard(deviceName='defaultKeyboard')
    text_space_end = visual.TextStim(win=win, name='text_space_end',
        text='Press SPACE to exit',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=(-0.14, -0.14, -0.10), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    # Run 'Begin Experiment' code from code_2
    marker = f"END"
    outlet.push_sample([marker])
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "lab_recorder" ---
    # create an object to store info about Routine lab_recorder
    lab_recorder = data.Routine(
        name='lab_recorder',
        components=[text_labrecorder, key_resp_labrecorder, text_space_labrecorder],
    )
    lab_recorder.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_labrecorder
    key_resp_labrecorder.keys = []
    key_resp_labrecorder.rt = []
    _key_resp_labrecorder_allKeys = []
    # store start times for lab_recorder
    lab_recorder.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    lab_recorder.tStart = globalClock.getTime(format='float')
    lab_recorder.status = STARTED
    thisExp.addData('lab_recorder.started', lab_recorder.tStart)
    lab_recorder.maxDuration = None
    # keep track of which components have finished
    lab_recorderComponents = lab_recorder.components
    for thisComponent in lab_recorder.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "lab_recorder" ---
    thisExp.currentRoutine = lab_recorder
    lab_recorder.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_labrecorder* updates
        
        # if text_labrecorder is starting this frame...
        if text_labrecorder.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_labrecorder.frameNStart = frameN  # exact frame index
            text_labrecorder.tStart = t  # local t and not account for scr refresh
            text_labrecorder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_labrecorder, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_labrecorder.status = STARTED
            text_labrecorder.setAutoDraw(True)
        
        # if text_labrecorder is active this frame...
        if text_labrecorder.status == STARTED:
            # update params
            pass
        
        # *key_resp_labrecorder* updates
        
        # if key_resp_labrecorder is starting this frame...
        if key_resp_labrecorder.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_labrecorder.frameNStart = frameN  # exact frame index
            key_resp_labrecorder.tStart = t  # local t and not account for scr refresh
            key_resp_labrecorder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_labrecorder, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_labrecorder.status = STARTED
            # keyboard checking is just starting
            key_resp_labrecorder.clock.reset()  # now t=0
        if key_resp_labrecorder.status == STARTED:
            theseKeys = key_resp_labrecorder.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_labrecorder_allKeys.extend(theseKeys)
            if len(_key_resp_labrecorder_allKeys):
                key_resp_labrecorder.keys = _key_resp_labrecorder_allKeys[-1].name  # just the last key pressed
                key_resp_labrecorder.rt = _key_resp_labrecorder_allKeys[-1].rt
                key_resp_labrecorder.duration = _key_resp_labrecorder_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_labrecorder* updates
        
        # if text_space_labrecorder is starting this frame...
        if text_space_labrecorder.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_labrecorder.frameNStart = frameN  # exact frame index
            text_space_labrecorder.tStart = t  # local t and not account for scr refresh
            text_space_labrecorder.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_labrecorder, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_labrecorder.status = STARTED
            text_space_labrecorder.setAutoDraw(True)
        
        # if text_space_labrecorder is active this frame...
        if text_space_labrecorder.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=lab_recorder,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            lab_recorder.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if lab_recorder.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in lab_recorder.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "lab_recorder" ---
    for thisComponent in lab_recorder.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for lab_recorder
    lab_recorder.tStop = globalClock.getTime(format='float')
    lab_recorder.tStopRefresh = tThisFlipGlobal
    thisExp.addData('lab_recorder.stopped', lab_recorder.tStop)
    thisExp.nextEntry()
    # the Routine "lab_recorder" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[Welcome, key_resp_welcome, text_space_welcome],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_welcome
    key_resp_welcome.keys = []
    key_resp_welcome.rt = []
    _key_resp_welcome_allKeys = []
    # Run 'Begin Routine' code from code
    marker = "WELCOME"
    outlet.push_sample([marker])
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    thisExp.currentRoutine = welcome
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *Welcome* updates
        
        # if Welcome is starting this frame...
        if Welcome.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            Welcome.frameNStart = frameN  # exact frame index
            Welcome.tStart = t  # local t and not account for scr refresh
            Welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(Welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            Welcome.status = STARTED
            Welcome.setAutoDraw(True)
        
        # if Welcome is active this frame...
        if Welcome.status == STARTED:
            # update params
            pass
        
        # *key_resp_welcome* updates
        
        # if key_resp_welcome is starting this frame...
        if key_resp_welcome.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_welcome.frameNStart = frameN  # exact frame index
            key_resp_welcome.tStart = t  # local t and not account for scr refresh
            key_resp_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_welcome.status = STARTED
            # keyboard checking is just starting
            key_resp_welcome.clock.reset()  # now t=0
        if key_resp_welcome.status == STARTED:
            theseKeys = key_resp_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_welcome_allKeys.extend(theseKeys)
            if len(_key_resp_welcome_allKeys):
                key_resp_welcome.keys = _key_resp_welcome_allKeys[-1].name  # just the last key pressed
                key_resp_welcome.rt = _key_resp_welcome_allKeys[-1].rt
                key_resp_welcome.duration = _key_resp_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_welcome* updates
        
        # if text_space_welcome is starting this frame...
        if text_space_welcome.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_welcome.frameNStart = frameN  # exact frame index
            text_space_welcome.tStart = t  # local t and not account for scr refresh
            text_space_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_welcome, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_welcome.status = STARTED
            text_space_welcome.setAutoDraw(True)
        
        # if text_space_welcome is active this frame...
        if text_space_welcome.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=welcome,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            welcome.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if welcome.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_instructions_closed" ---
    # create an object to store info about Routine calibration_instructions_closed
    calibration_instructions_closed = data.Routine(
        name='calibration_instructions_closed',
        components=[text_calibration_instructions_closed, key_resp_calibration_instructions_closed, text_space_calibration_instructions_closed],
    )
    calibration_instructions_closed.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_calibration_instructions_closed
    key_resp_calibration_instructions_closed.keys = []
    key_resp_calibration_instructions_closed.rt = []
    _key_resp_calibration_instructions_closed_allKeys = []
    # store start times for calibration_instructions_closed
    calibration_instructions_closed.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_instructions_closed.tStart = globalClock.getTime(format='float')
    calibration_instructions_closed.status = STARTED
    thisExp.addData('calibration_instructions_closed.started', calibration_instructions_closed.tStart)
    calibration_instructions_closed.maxDuration = None
    # keep track of which components have finished
    calibration_instructions_closedComponents = calibration_instructions_closed.components
    for thisComponent in calibration_instructions_closed.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_instructions_closed" ---
    thisExp.currentRoutine = calibration_instructions_closed
    calibration_instructions_closed.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_calibration_instructions_closed* updates
        
        # if text_calibration_instructions_closed is starting this frame...
        if text_calibration_instructions_closed.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_calibration_instructions_closed.frameNStart = frameN  # exact frame index
            text_calibration_instructions_closed.tStart = t  # local t and not account for scr refresh
            text_calibration_instructions_closed.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_calibration_instructions_closed, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_calibration_instructions_closed.status = STARTED
            text_calibration_instructions_closed.setAutoDraw(True)
        
        # if text_calibration_instructions_closed is active this frame...
        if text_calibration_instructions_closed.status == STARTED:
            # update params
            pass
        
        # *key_resp_calibration_instructions_closed* updates
        
        # if key_resp_calibration_instructions_closed is starting this frame...
        if key_resp_calibration_instructions_closed.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_calibration_instructions_closed.frameNStart = frameN  # exact frame index
            key_resp_calibration_instructions_closed.tStart = t  # local t and not account for scr refresh
            key_resp_calibration_instructions_closed.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_calibration_instructions_closed, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_calibration_instructions_closed.status = STARTED
            # keyboard checking is just starting
            key_resp_calibration_instructions_closed.clock.reset()  # now t=0
        if key_resp_calibration_instructions_closed.status == STARTED:
            theseKeys = key_resp_calibration_instructions_closed.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_calibration_instructions_closed_allKeys.extend(theseKeys)
            if len(_key_resp_calibration_instructions_closed_allKeys):
                key_resp_calibration_instructions_closed.keys = _key_resp_calibration_instructions_closed_allKeys[-1].name  # just the last key pressed
                key_resp_calibration_instructions_closed.rt = _key_resp_calibration_instructions_closed_allKeys[-1].rt
                key_resp_calibration_instructions_closed.duration = _key_resp_calibration_instructions_closed_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_calibration_instructions_closed* updates
        
        # if text_space_calibration_instructions_closed is starting this frame...
        if text_space_calibration_instructions_closed.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_calibration_instructions_closed.frameNStart = frameN  # exact frame index
            text_space_calibration_instructions_closed.tStart = t  # local t and not account for scr refresh
            text_space_calibration_instructions_closed.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_calibration_instructions_closed, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_calibration_instructions_closed.status = STARTED
            text_space_calibration_instructions_closed.setAutoDraw(True)
        
        # if text_space_calibration_instructions_closed is active this frame...
        if text_space_calibration_instructions_closed.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=calibration_instructions_closed,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            calibration_instructions_closed.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if calibration_instructions_closed.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in calibration_instructions_closed.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_instructions_closed" ---
    for thisComponent in calibration_instructions_closed.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_instructions_closed
    calibration_instructions_closed.tStop = globalClock.getTime(format='float')
    calibration_instructions_closed.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_instructions_closed.stopped', calibration_instructions_closed.tStop)
    thisExp.nextEntry()
    # the Routine "calibration_instructions_closed" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_time_closed" ---
    # create an object to store info about Routine calibration_time_closed
    calibration_time_closed = data.Routine(
        name='calibration_time_closed',
        components=[text],
    )
    calibration_time_closed.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_time_closed
    marker = "CALIBRATION_START_EYES_CLOSED"
    outlet.push_sample([marker])
    # store start times for calibration_time_closed
    calibration_time_closed.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_time_closed.tStart = globalClock.getTime(format='float')
    calibration_time_closed.status = STARTED
    thisExp.addData('calibration_time_closed.started', calibration_time_closed.tStart)
    calibration_time_closed.maxDuration = 60
    # keep track of which components have finished
    calibration_time_closedComponents = calibration_time_closed.components
    for thisComponent in calibration_time_closed.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_time_closed" ---
    thisExp.currentRoutine = calibration_time_closed
    calibration_time_closed.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > calibration_time_closed.maxDuration-frameTolerance:
            calibration_time_closed.maxDurationReached = True
            continueRoutine = False
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # if text is stopping this frame...
        if text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text.tStop = t  # not accounting for scr refresh
                text.tStopRefresh = tThisFlipGlobal  # on global time
                text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.stopped')
                # update status
                text.status = FINISHED
                text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=calibration_time_closed,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            calibration_time_closed.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if calibration_time_closed.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in calibration_time_closed.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_time_closed" ---
    for thisComponent in calibration_time_closed.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_time_closed
    calibration_time_closed.tStop = globalClock.getTime(format='float')
    calibration_time_closed.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_time_closed.stopped', calibration_time_closed.tStop)
    # Run 'End Routine' code from code_time_closed
    marker = "CALIBRATION_END_EYES_CLOSED"
    outlet.push_sample([marker])
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if calibration_time_closed.maxDurationReached:
        routineTimer.addTime(-calibration_time_closed.maxDuration)
    elif calibration_time_closed.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "calibration_instructions_open" ---
    # create an object to store info about Routine calibration_instructions_open
    calibration_instructions_open = data.Routine(
        name='calibration_instructions_open',
        components=[text_calibration_instructions_open, key_resp_calibration_instructions_open, text_space_calibration_instructions_open, sound_closed],
    )
    calibration_instructions_open.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_calibration_instructions_open
    key_resp_calibration_instructions_open.keys = []
    key_resp_calibration_instructions_open.rt = []
    _key_resp_calibration_instructions_open_allKeys = []
    sound_closed.setSound('A', secs=0.5, hamming=True)
    sound_closed.setVolume(1.0, log=False)
    sound_closed.seek(0)
    # store start times for calibration_instructions_open
    calibration_instructions_open.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_instructions_open.tStart = globalClock.getTime(format='float')
    calibration_instructions_open.status = STARTED
    thisExp.addData('calibration_instructions_open.started', calibration_instructions_open.tStart)
    calibration_instructions_open.maxDuration = None
    # keep track of which components have finished
    calibration_instructions_openComponents = calibration_instructions_open.components
    for thisComponent in calibration_instructions_open.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_instructions_open" ---
    thisExp.currentRoutine = calibration_instructions_open
    calibration_instructions_open.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_calibration_instructions_open* updates
        
        # if text_calibration_instructions_open is starting this frame...
        if text_calibration_instructions_open.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_calibration_instructions_open.frameNStart = frameN  # exact frame index
            text_calibration_instructions_open.tStart = t  # local t and not account for scr refresh
            text_calibration_instructions_open.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_calibration_instructions_open, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_calibration_instructions_open.status = STARTED
            text_calibration_instructions_open.setAutoDraw(True)
        
        # if text_calibration_instructions_open is active this frame...
        if text_calibration_instructions_open.status == STARTED:
            # update params
            pass
        
        # *key_resp_calibration_instructions_open* updates
        
        # if key_resp_calibration_instructions_open is starting this frame...
        if key_resp_calibration_instructions_open.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_calibration_instructions_open.frameNStart = frameN  # exact frame index
            key_resp_calibration_instructions_open.tStart = t  # local t and not account for scr refresh
            key_resp_calibration_instructions_open.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_calibration_instructions_open, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_calibration_instructions_open.status = STARTED
            # keyboard checking is just starting
            key_resp_calibration_instructions_open.clock.reset()  # now t=0
        if key_resp_calibration_instructions_open.status == STARTED:
            theseKeys = key_resp_calibration_instructions_open.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_calibration_instructions_open_allKeys.extend(theseKeys)
            if len(_key_resp_calibration_instructions_open_allKeys):
                key_resp_calibration_instructions_open.keys = _key_resp_calibration_instructions_open_allKeys[-1].name  # just the last key pressed
                key_resp_calibration_instructions_open.rt = _key_resp_calibration_instructions_open_allKeys[-1].rt
                key_resp_calibration_instructions_open.duration = _key_resp_calibration_instructions_open_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_calibration_instructions_open* updates
        
        # if text_space_calibration_instructions_open is starting this frame...
        if text_space_calibration_instructions_open.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_calibration_instructions_open.frameNStart = frameN  # exact frame index
            text_space_calibration_instructions_open.tStart = t  # local t and not account for scr refresh
            text_space_calibration_instructions_open.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_calibration_instructions_open, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_calibration_instructions_open.status = STARTED
            text_space_calibration_instructions_open.setAutoDraw(True)
        
        # if text_space_calibration_instructions_open is active this frame...
        if text_space_calibration_instructions_open.status == STARTED:
            # update params
            pass
        
        # *sound_closed* updates
        
        # if sound_closed is starting this frame...
        if sound_closed.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sound_closed.frameNStart = frameN  # exact frame index
            sound_closed.tStart = t  # local t and not account for scr refresh
            sound_closed.tStartRefresh = tThisFlipGlobal  # on global time
            # add timestamp to datafile
            thisExp.addData('sound_closed.started', tThisFlipGlobal)
            # update status
            sound_closed.status = STARTED
            sound_closed.play(when=win)  # sync with win flip
        
        # if sound_closed is stopping this frame...
        if sound_closed.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > sound_closed.tStartRefresh + 0.5-frameTolerance or sound_closed.isFinished:
                # keep track of stop time/frame for later
                sound_closed.tStop = t  # not accounting for scr refresh
                sound_closed.tStopRefresh = tThisFlipGlobal  # on global time
                sound_closed.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'sound_closed.stopped')
                # update status
                sound_closed.status = FINISHED
                sound_closed.stop()
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=calibration_instructions_open,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            calibration_instructions_open.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if calibration_instructions_open.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in calibration_instructions_open.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_instructions_open" ---
    for thisComponent in calibration_instructions_open.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_instructions_open
    calibration_instructions_open.tStop = globalClock.getTime(format='float')
    calibration_instructions_open.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_instructions_open.stopped', calibration_instructions_open.tStop)
    sound_closed.pause()  # ensure sound has stopped at end of Routine
    thisExp.nextEntry()
    # the Routine "calibration_instructions_open" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "calibration_time_open" ---
    # create an object to store info about Routine calibration_time_open
    calibration_time_open = data.Routine(
        name='calibration_time_open',
        components=[text_2],
    )
    calibration_time_open.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_time_open
    marker = "CALIBRATION_START_EYES_OPEN"
    outlet.push_sample([marker])
    # store start times for calibration_time_open
    calibration_time_open.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    calibration_time_open.tStart = globalClock.getTime(format='float')
    calibration_time_open.status = STARTED
    thisExp.addData('calibration_time_open.started', calibration_time_open.tStart)
    calibration_time_open.maxDuration = 60
    # keep track of which components have finished
    calibration_time_openComponents = calibration_time_open.components
    for thisComponent in calibration_time_open.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "calibration_time_open" ---
    thisExp.currentRoutine = calibration_time_open
    calibration_time_open.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 60.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > calibration_time_open.maxDuration-frameTolerance:
            calibration_time_open.maxDurationReached = True
            continueRoutine = False
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 60-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=calibration_time_open,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            calibration_time_open.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if calibration_time_open.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in calibration_time_open.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "calibration_time_open" ---
    for thisComponent in calibration_time_open.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for calibration_time_open
    calibration_time_open.tStop = globalClock.getTime(format='float')
    calibration_time_open.tStopRefresh = tThisFlipGlobal
    thisExp.addData('calibration_time_open.stopped', calibration_time_open.tStop)
    # Run 'End Routine' code from code_time_open
    marker = "CALIBRATION_END_EYES_OPEN"
    outlet.push_sample([marker])
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if calibration_time_open.maxDurationReached:
        routineTimer.addTime(-calibration_time_open.maxDuration)
    elif calibration_time_open.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-60.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instructions_willingness" ---
    # create an object to store info about Routine instructions_willingness
    instructions_willingness = data.Routine(
        name='instructions_willingness',
        components=[text_instructions_willingness, key_resp_instructions_willingness, text_space_instructions_willingness],
    )
    instructions_willingness.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instructions_willingness
    key_resp_instructions_willingness.keys = []
    key_resp_instructions_willingness.rt = []
    _key_resp_instructions_willingness_allKeys = []
    # store start times for instructions_willingness
    instructions_willingness.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_willingness.tStart = globalClock.getTime(format='float')
    instructions_willingness.status = STARTED
    thisExp.addData('instructions_willingness.started', instructions_willingness.tStart)
    instructions_willingness.maxDuration = None
    # keep track of which components have finished
    instructions_willingnessComponents = instructions_willingness.components
    for thisComponent in instructions_willingness.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_willingness" ---
    thisExp.currentRoutine = instructions_willingness
    instructions_willingness.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_willingness* updates
        
        # if text_instructions_willingness is starting this frame...
        if text_instructions_willingness.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_willingness.frameNStart = frameN  # exact frame index
            text_instructions_willingness.tStart = t  # local t and not account for scr refresh
            text_instructions_willingness.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_willingness, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instructions_willingness.status = STARTED
            text_instructions_willingness.setAutoDraw(True)
        
        # if text_instructions_willingness is active this frame...
        if text_instructions_willingness.status == STARTED:
            # update params
            pass
        
        # *key_resp_instructions_willingness* updates
        
        # if key_resp_instructions_willingness is starting this frame...
        if key_resp_instructions_willingness.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instructions_willingness.frameNStart = frameN  # exact frame index
            key_resp_instructions_willingness.tStart = t  # local t and not account for scr refresh
            key_resp_instructions_willingness.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instructions_willingness, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_instructions_willingness.status = STARTED
            # keyboard checking is just starting
            key_resp_instructions_willingness.clock.reset()  # now t=0
        if key_resp_instructions_willingness.status == STARTED:
            theseKeys = key_resp_instructions_willingness.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instructions_willingness_allKeys.extend(theseKeys)
            if len(_key_resp_instructions_willingness_allKeys):
                key_resp_instructions_willingness.keys = _key_resp_instructions_willingness_allKeys[-1].name  # just the last key pressed
                key_resp_instructions_willingness.rt = _key_resp_instructions_willingness_allKeys[-1].rt
                key_resp_instructions_willingness.duration = _key_resp_instructions_willingness_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_instructions_willingness* updates
        
        # if text_space_instructions_willingness is starting this frame...
        if text_space_instructions_willingness.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_instructions_willingness.frameNStart = frameN  # exact frame index
            text_space_instructions_willingness.tStart = t  # local t and not account for scr refresh
            text_space_instructions_willingness.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_instructions_willingness, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_instructions_willingness.status = STARTED
            text_space_instructions_willingness.setAutoDraw(True)
        
        # if text_space_instructions_willingness is active this frame...
        if text_space_instructions_willingness.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_willingness,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_willingness.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_willingness.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_willingness.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_willingness" ---
    for thisComponent in instructions_willingness.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_willingness
    instructions_willingness.tStop = globalClock.getTime(format='float')
    instructions_willingness.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_willingness.stopped', instructions_willingness.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_willingness" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    willingness = data.TrialHandler2(
        name='willingness',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('willingness.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(willingness)  # add the loop to the experiment
    thisWillingnes = willingness.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisWillingnes.rgb)
    if thisWillingnes != None:
        for paramName in thisWillingnes:
            globals()[paramName] = thisWillingnes[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisWillingnes in willingness:
        willingness.status = STARTED
        if hasattr(thisWillingnes, 'status'):
            thisWillingnes.status = STARTED
        currentLoop = willingness
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisWillingnes.rgb)
        if thisWillingnes != None:
            for paramName in thisWillingnes:
                globals()[paramName] = thisWillingnes[paramName]
        
        # --- Prepare to start Routine "test_willingness" ---
        # create an object to store info about Routine test_willingness
        test_willingness = data.Routine(
            name='test_willingness',
            components=[text_willingness, slider_willingness, mouse_willingness, prog_willingness],
        )
        test_willingness.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_willingness.setText(Question)
        slider_willingness.reset()
        # setup some python lists for storing info about the mouse_willingness
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_willingness
        mouse_willingness.clickReset()
        prevButtonState = mouse_willingness.getPressed()[0]
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_willingness.setProgress(willingness.thisN / 6)
        # store start times for test_willingness
        test_willingness.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_willingness.tStart = globalClock.getTime(format='float')
        test_willingness.status = STARTED
        thisExp.addData('test_willingness.started', test_willingness.tStart)
        test_willingness.maxDuration = None
        # keep track of which components have finished
        test_willingnessComponents = test_willingness.components
        for thisComponent in test_willingness.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_willingness" ---
        thisExp.currentRoutine = test_willingness
        test_willingness.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisWillingnes, 'status') and thisWillingnes.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_willingness* updates
            
            # if text_willingness is starting this frame...
            if text_willingness.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_willingness.frameNStart = frameN  # exact frame index
                text_willingness.tStart = t  # local t and not account for scr refresh
                text_willingness.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_willingness, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_willingness.status = STARTED
                text_willingness.setAutoDraw(True)
            
            # if text_willingness is active this frame...
            if text_willingness.status == STARTED:
                # update params
                pass
            
            # *slider_willingness* updates
            
            # if slider_willingness is starting this frame...
            if slider_willingness.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_willingness.frameNStart = frameN  # exact frame index
                slider_willingness.tStart = t  # local t and not account for scr refresh
                slider_willingness.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_willingness, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_willingness.started')
                # update status
                slider_willingness.status = STARTED
                slider_willingness.setAutoDraw(True)
            
            # if slider_willingness is active this frame...
            if slider_willingness.status == STARTED:
                # update params
                pass
            # *mouse_willingness* updates
            
            # if mouse_willingness is starting this frame...
            if mouse_willingness.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_willingness.frameNStart = frameN  # exact frame index
                mouse_willingness.tStart = t  # local t and not account for scr refresh
                mouse_willingness.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_willingness, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_willingness.status = STARTED
                mouse_willingness.mouseClock.reset()
                prevButtonState = mouse_willingness.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_willingness
            mouse_pos = mouse_willingness.getPos()
            
            x_min = slider_willingness.pos[0] - slider_willingness.size[0] / 2
            x_max = slider_willingness.pos[0] + slider_willingness.size[0] / 2
            y_min = slider_willingness.pos[1] - 0.1
            y_max = slider_willingness.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_willingness.markerPos = slider_willingness._posToRating(mouse_pos)
            
            buttonState = mouse_willingness.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_willingness.rating = slider_willingness.markerPos
                    continueRoutine = False
            
            prevButtonState = buttonState
            
            # *prog_willingness* updates
            
            # if prog_willingness is starting this frame...
            if prog_willingness.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_willingness.frameNStart = frameN  # exact frame index
                prog_willingness.tStart = t  # local t and not account for scr refresh
                prog_willingness.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_willingness, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_willingness.started')
                # update status
                prog_willingness.status = STARTED
                prog_willingness.setAutoDraw(True)
            
            # if prog_willingness is active this frame...
            if prog_willingness.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_willingness,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                test_willingness.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if test_willingness.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in test_willingness.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_willingness" ---
        for thisComponent in test_willingness.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_willingness
        test_willingness.tStop = globalClock.getTime(format='float')
        test_willingness.tStopRefresh = tThisFlipGlobal
        thisExp.addData('test_willingness.stopped', test_willingness.tStop)
        # store data for willingness (TrialHandler)
        # Run 'End Routine' code from code_willingness
        thisExp.addData('trial', willingness.thisN + 1)
        thisExp.addData('answ_willingness', slider_willingness.markerPos)
        # the Routine "test_willingness" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisWillingnes as finished
        if hasattr(thisWillingnes, 'status'):
            thisWillingnes.status = FINISHED
        # if awaiting a pause, pause now
        if willingness.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            willingness.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'willingness'
    willingness.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "instructions_trial" ---
    # create an object to store info about Routine instructions_trial
    instructions_trial = data.Routine(
        name='instructions_trial',
        components=[text_instructions_trial, key_resp_instructions_trial, text_space_instructions_trial],
    )
    instructions_trial.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instructions_trial
    key_resp_instructions_trial.keys = []
    key_resp_instructions_trial.rt = []
    _key_resp_instructions_trial_allKeys = []
    # store start times for instructions_trial
    instructions_trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_trial.tStart = globalClock.getTime(format='float')
    instructions_trial.status = STARTED
    thisExp.addData('instructions_trial.started', instructions_trial.tStart)
    instructions_trial.maxDuration = None
    # keep track of which components have finished
    instructions_trialComponents = instructions_trial.components
    for thisComponent in instructions_trial.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_trial" ---
    thisExp.currentRoutine = instructions_trial
    instructions_trial.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_trial* updates
        
        # if text_instructions_trial is starting this frame...
        if text_instructions_trial.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_trial.frameNStart = frameN  # exact frame index
            text_instructions_trial.tStart = t  # local t and not account for scr refresh
            text_instructions_trial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_trial, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instructions_trial.status = STARTED
            text_instructions_trial.setAutoDraw(True)
        
        # if text_instructions_trial is active this frame...
        if text_instructions_trial.status == STARTED:
            # update params
            pass
        
        # *key_resp_instructions_trial* updates
        
        # if key_resp_instructions_trial is starting this frame...
        if key_resp_instructions_trial.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instructions_trial.frameNStart = frameN  # exact frame index
            key_resp_instructions_trial.tStart = t  # local t and not account for scr refresh
            key_resp_instructions_trial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instructions_trial, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_instructions_trial.status = STARTED
            # keyboard checking is just starting
            key_resp_instructions_trial.clock.reset()  # now t=0
        if key_resp_instructions_trial.status == STARTED:
            theseKeys = key_resp_instructions_trial.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instructions_trial_allKeys.extend(theseKeys)
            if len(_key_resp_instructions_trial_allKeys):
                key_resp_instructions_trial.keys = _key_resp_instructions_trial_allKeys[-1].name  # just the last key pressed
                key_resp_instructions_trial.rt = _key_resp_instructions_trial_allKeys[-1].rt
                key_resp_instructions_trial.duration = _key_resp_instructions_trial_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_instructions_trial* updates
        
        # if text_space_instructions_trial is starting this frame...
        if text_space_instructions_trial.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_instructions_trial.frameNStart = frameN  # exact frame index
            text_space_instructions_trial.tStart = t  # local t and not account for scr refresh
            text_space_instructions_trial.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_instructions_trial, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_instructions_trial.status = STARTED
            text_space_instructions_trial.setAutoDraw(True)
        
        # if text_space_instructions_trial is active this frame...
        if text_space_instructions_trial.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_trial,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_trial.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_trial.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_trial.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_trial" ---
    for thisComponent in instructions_trial.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_trial
    instructions_trial.tStop = globalClock.getTime(format='float')
    instructions_trial.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_trial.stopped', instructions_trial.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_trial_2" ---
    # create an object to store info about Routine instructions_trial_2
    instructions_trial_2 = data.Routine(
        name='instructions_trial_2',
        components=[key_resp_instructions_trial_2, text_space_instructions_trial_2, text_instructions_trial_2],
    )
    instructions_trial_2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instructions_trial_2
    key_resp_instructions_trial_2.keys = []
    key_resp_instructions_trial_2.rt = []
    _key_resp_instructions_trial_2_allKeys = []
    # store start times for instructions_trial_2
    instructions_trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_trial_2.tStart = globalClock.getTime(format='float')
    instructions_trial_2.status = STARTED
    thisExp.addData('instructions_trial_2.started', instructions_trial_2.tStart)
    instructions_trial_2.maxDuration = None
    # keep track of which components have finished
    instructions_trial_2Components = instructions_trial_2.components
    for thisComponent in instructions_trial_2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_trial_2" ---
    thisExp.currentRoutine = instructions_trial_2
    instructions_trial_2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_instructions_trial_2* updates
        
        # if key_resp_instructions_trial_2 is starting this frame...
        if key_resp_instructions_trial_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instructions_trial_2.frameNStart = frameN  # exact frame index
            key_resp_instructions_trial_2.tStart = t  # local t and not account for scr refresh
            key_resp_instructions_trial_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instructions_trial_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_instructions_trial_2.status = STARTED
            # keyboard checking is just starting
            key_resp_instructions_trial_2.clock.reset()  # now t=0
        if key_resp_instructions_trial_2.status == STARTED:
            theseKeys = key_resp_instructions_trial_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instructions_trial_2_allKeys.extend(theseKeys)
            if len(_key_resp_instructions_trial_2_allKeys):
                key_resp_instructions_trial_2.keys = _key_resp_instructions_trial_2_allKeys[-1].name  # just the last key pressed
                key_resp_instructions_trial_2.rt = _key_resp_instructions_trial_2_allKeys[-1].rt
                key_resp_instructions_trial_2.duration = _key_resp_instructions_trial_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_instructions_trial_2* updates
        
        # if text_space_instructions_trial_2 is starting this frame...
        if text_space_instructions_trial_2.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_instructions_trial_2.frameNStart = frameN  # exact frame index
            text_space_instructions_trial_2.tStart = t  # local t and not account for scr refresh
            text_space_instructions_trial_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_instructions_trial_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_instructions_trial_2.status = STARTED
            text_space_instructions_trial_2.setAutoDraw(True)
        
        # if text_space_instructions_trial_2 is active this frame...
        if text_space_instructions_trial_2.status == STARTED:
            # update params
            pass
        
        # *text_instructions_trial_2* updates
        
        # if text_instructions_trial_2 is starting this frame...
        if text_instructions_trial_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_trial_2.frameNStart = frameN  # exact frame index
            text_instructions_trial_2.tStart = t  # local t and not account for scr refresh
            text_instructions_trial_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_trial_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instructions_trial_2.status = STARTED
            text_instructions_trial_2.setAutoDraw(True)
        
        # if text_instructions_trial_2 is active this frame...
        if text_instructions_trial_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_trial_2,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_trial_2.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_trial_2.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_trial_2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_trial_2" ---
    for thisComponent in instructions_trial_2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_trial_2
    instructions_trial_2.tStop = globalClock.getTime(format='float')
    instructions_trial_2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_trial_2.stopped', instructions_trial_2.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_trial_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_trial_3" ---
    # create an object to store info about Routine instructions_trial_3
    instructions_trial_3 = data.Routine(
        name='instructions_trial_3',
        components=[text_instrucciones_ejercicio__3, key_resp_instrucciones_ejercicio_3, text_space_instrucciones_ejercicio_3],
    )
    instructions_trial_3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instrucciones_ejercicio_3
    key_resp_instrucciones_ejercicio_3.keys = []
    key_resp_instrucciones_ejercicio_3.rt = []
    _key_resp_instrucciones_ejercicio_3_allKeys = []
    # store start times for instructions_trial_3
    instructions_trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_trial_3.tStart = globalClock.getTime(format='float')
    instructions_trial_3.status = STARTED
    thisExp.addData('instructions_trial_3.started', instructions_trial_3.tStart)
    instructions_trial_3.maxDuration = None
    # keep track of which components have finished
    instructions_trial_3Components = instructions_trial_3.components
    for thisComponent in instructions_trial_3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_trial_3" ---
    thisExp.currentRoutine = instructions_trial_3
    instructions_trial_3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instrucciones_ejercicio__3* updates
        
        # if text_instrucciones_ejercicio__3 is starting this frame...
        if text_instrucciones_ejercicio__3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instrucciones_ejercicio__3.frameNStart = frameN  # exact frame index
            text_instrucciones_ejercicio__3.tStart = t  # local t and not account for scr refresh
            text_instrucciones_ejercicio__3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instrucciones_ejercicio__3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instrucciones_ejercicio__3.status = STARTED
            text_instrucciones_ejercicio__3.setAutoDraw(True)
        
        # if text_instrucciones_ejercicio__3 is active this frame...
        if text_instrucciones_ejercicio__3.status == STARTED:
            # update params
            pass
        
        # *key_resp_instrucciones_ejercicio_3* updates
        
        # if key_resp_instrucciones_ejercicio_3 is starting this frame...
        if key_resp_instrucciones_ejercicio_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instrucciones_ejercicio_3.frameNStart = frameN  # exact frame index
            key_resp_instrucciones_ejercicio_3.tStart = t  # local t and not account for scr refresh
            key_resp_instrucciones_ejercicio_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instrucciones_ejercicio_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_instrucciones_ejercicio_3.status = STARTED
            # keyboard checking is just starting
            key_resp_instrucciones_ejercicio_3.clock.reset()  # now t=0
        if key_resp_instrucciones_ejercicio_3.status == STARTED:
            theseKeys = key_resp_instrucciones_ejercicio_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instrucciones_ejercicio_3_allKeys.extend(theseKeys)
            if len(_key_resp_instrucciones_ejercicio_3_allKeys):
                key_resp_instrucciones_ejercicio_3.keys = _key_resp_instrucciones_ejercicio_3_allKeys[-1].name  # just the last key pressed
                key_resp_instrucciones_ejercicio_3.rt = _key_resp_instrucciones_ejercicio_3_allKeys[-1].rt
                key_resp_instrucciones_ejercicio_3.duration = _key_resp_instrucciones_ejercicio_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_instrucciones_ejercicio_3* updates
        
        # if text_space_instrucciones_ejercicio_3 is starting this frame...
        if text_space_instrucciones_ejercicio_3.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_instrucciones_ejercicio_3.frameNStart = frameN  # exact frame index
            text_space_instrucciones_ejercicio_3.tStart = t  # local t and not account for scr refresh
            text_space_instrucciones_ejercicio_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_instrucciones_ejercicio_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_instrucciones_ejercicio_3.status = STARTED
            text_space_instrucciones_ejercicio_3.setAutoDraw(True)
        
        # if text_space_instrucciones_ejercicio_3 is active this frame...
        if text_space_instrucciones_ejercicio_3.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_trial_3,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_trial_3.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_trial_3.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_trial_3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_trial_3" ---
    for thisComponent in instructions_trial_3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_trial_3
    instructions_trial_3.tStop = globalClock.getTime(format='float')
    instructions_trial_3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_trial_3.stopped', instructions_trial_3.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_trial_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_trial_4" ---
    # create an object to store info about Routine instructions_trial_4
    instructions_trial_4 = data.Routine(
        name='instructions_trial_4',
        components=[text_instructions_trial_4, key_resp_instructions_trial_4, text_space_instructions_trial_4],
    )
    instructions_trial_4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_instructions_trial_4
    key_resp_instructions_trial_4.keys = []
    key_resp_instructions_trial_4.rt = []
    _key_resp_instructions_trial_4_allKeys = []
    # store start times for instructions_trial_4
    instructions_trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_trial_4.tStart = globalClock.getTime(format='float')
    instructions_trial_4.status = STARTED
    thisExp.addData('instructions_trial_4.started', instructions_trial_4.tStart)
    instructions_trial_4.maxDuration = None
    # keep track of which components have finished
    instructions_trial_4Components = instructions_trial_4.components
    for thisComponent in instructions_trial_4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_trial_4" ---
    thisExp.currentRoutine = instructions_trial_4
    instructions_trial_4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instructions_trial_4* updates
        
        # if text_instructions_trial_4 is starting this frame...
        if text_instructions_trial_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instructions_trial_4.frameNStart = frameN  # exact frame index
            text_instructions_trial_4.tStart = t  # local t and not account for scr refresh
            text_instructions_trial_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instructions_trial_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_instructions_trial_4.status = STARTED
            text_instructions_trial_4.setAutoDraw(True)
        
        # if text_instructions_trial_4 is active this frame...
        if text_instructions_trial_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_instructions_trial_4* updates
        
        # if key_resp_instructions_trial_4 is starting this frame...
        if key_resp_instructions_trial_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_instructions_trial_4.frameNStart = frameN  # exact frame index
            key_resp_instructions_trial_4.tStart = t  # local t and not account for scr refresh
            key_resp_instructions_trial_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_instructions_trial_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_instructions_trial_4.status = STARTED
            # keyboard checking is just starting
            key_resp_instructions_trial_4.clock.reset()  # now t=0
        if key_resp_instructions_trial_4.status == STARTED:
            theseKeys = key_resp_instructions_trial_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_instructions_trial_4_allKeys.extend(theseKeys)
            if len(_key_resp_instructions_trial_4_allKeys):
                key_resp_instructions_trial_4.keys = _key_resp_instructions_trial_4_allKeys[-1].name  # just the last key pressed
                key_resp_instructions_trial_4.rt = _key_resp_instructions_trial_4_allKeys[-1].rt
                key_resp_instructions_trial_4.duration = _key_resp_instructions_trial_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_instructions_trial_4* updates
        
        # if text_space_instructions_trial_4 is starting this frame...
        if text_space_instructions_trial_4.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_instructions_trial_4.frameNStart = frameN  # exact frame index
            text_space_instructions_trial_4.tStart = t  # local t and not account for scr refresh
            text_space_instructions_trial_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_instructions_trial_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_instructions_trial_4.status = STARTED
            text_space_instructions_trial_4.setAutoDraw(True)
        
        # if text_space_instructions_trial_4 is active this frame...
        if text_space_instructions_trial_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_trial_4,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_trial_4.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_trial_4.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_trial_4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_trial_4" ---
    for thisComponent in instructions_trial_4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_trial_4
    instructions_trial_4.tStop = globalClock.getTime(format='float')
    instructions_trial_4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_trial_4.stopped', instructions_trial_4.tStop)
    thisExp.nextEntry()
    # the Routine "instructions_trial_4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "trial_manual_start" ---
    # create an object to store info about Routine trial_manual_start
    trial_manual_start = data.Routine(
        name='trial_manual_start',
        components=[text_trial_manual_start, key_resp_trial_manual_start, text_space_trial_manual_start],
    )
    trial_manual_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_trial_manual_start
    key_resp_trial_manual_start.keys = []
    key_resp_trial_manual_start.rt = []
    _key_resp_trial_manual_start_allKeys = []
    # store start times for trial_manual_start
    trial_manual_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    trial_manual_start.tStart = globalClock.getTime(format='float')
    trial_manual_start.status = STARTED
    thisExp.addData('trial_manual_start.started', trial_manual_start.tStart)
    trial_manual_start.maxDuration = None
    # keep track of which components have finished
    trial_manual_startComponents = trial_manual_start.components
    for thisComponent in trial_manual_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "trial_manual_start" ---
    thisExp.currentRoutine = trial_manual_start
    trial_manual_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_trial_manual_start* updates
        
        # if text_trial_manual_start is starting this frame...
        if text_trial_manual_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_trial_manual_start.frameNStart = frameN  # exact frame index
            text_trial_manual_start.tStart = t  # local t and not account for scr refresh
            text_trial_manual_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_trial_manual_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_trial_manual_start.status = STARTED
            text_trial_manual_start.setAutoDraw(True)
        
        # if text_trial_manual_start is active this frame...
        if text_trial_manual_start.status == STARTED:
            # update params
            pass
        
        # *key_resp_trial_manual_start* updates
        
        # if key_resp_trial_manual_start is starting this frame...
        if key_resp_trial_manual_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_trial_manual_start.frameNStart = frameN  # exact frame index
            key_resp_trial_manual_start.tStart = t  # local t and not account for scr refresh
            key_resp_trial_manual_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_trial_manual_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_trial_manual_start.status = STARTED
            # keyboard checking is just starting
            key_resp_trial_manual_start.clock.reset()  # now t=0
        if key_resp_trial_manual_start.status == STARTED:
            theseKeys = key_resp_trial_manual_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_trial_manual_start_allKeys.extend(theseKeys)
            if len(_key_resp_trial_manual_start_allKeys):
                key_resp_trial_manual_start.keys = _key_resp_trial_manual_start_allKeys[-1].name  # just the last key pressed
                key_resp_trial_manual_start.rt = _key_resp_trial_manual_start_allKeys[-1].rt
                key_resp_trial_manual_start.duration = _key_resp_trial_manual_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_trial_manual_start* updates
        
        # if text_space_trial_manual_start is starting this frame...
        if text_space_trial_manual_start.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_trial_manual_start.frameNStart = frameN  # exact frame index
            text_space_trial_manual_start.tStart = t  # local t and not account for scr refresh
            text_space_trial_manual_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_trial_manual_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_trial_manual_start.status = STARTED
            text_space_trial_manual_start.setAutoDraw(True)
        
        # if text_space_trial_manual_start is active this frame...
        if text_space_trial_manual_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=trial_manual_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            trial_manual_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if trial_manual_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in trial_manual_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "trial_manual_start" ---
    for thisComponent in trial_manual_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for trial_manual_start
    trial_manual_start.tStop = globalClock.getTime(format='float')
    trial_manual_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('trial_manual_start.stopped', trial_manual_start.tStop)
    thisExp.nextEntry()
    # the Routine "trial_manual_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    manual = data.TrialHandler2(
        name='manual',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trial_manual_ecological.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(manual)  # add the loop to the experiment
    thisManual = manual.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisManual.rgb)
    if thisManual != None:
        for paramName in thisManual:
            globals()[paramName] = thisManual[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisManual in manual:
        manual.status = STARTED
        if hasattr(thisManual, 'status'):
            thisManual.status = STARTED
        currentLoop = manual
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisManual.rgb)
        if thisManual != None:
            for paramName in thisManual:
                globals()[paramName] = thisManual[paramName]
        
        # --- Prepare to start Routine "time_manual" ---
        # create an object to store info about Routine time_manual
        time_manual = data.Routine(
            name='time_manual',
            components=[],
        )
        time_manual.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for time_manual
        time_manual.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_manual.tStart = globalClock.getTime(format='float')
        time_manual.status = STARTED
        thisExp.addData('time_manual.started', time_manual.tStart)
        time_manual.maxDuration = 1
        # keep track of which components have finished
        time_manualComponents = time_manual.components
        for thisComponent in time_manual.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "time_manual" ---
        thisExp.currentRoutine = time_manual
        time_manual.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisManual, 'status') and thisManual.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > time_manual.maxDuration-frameTolerance:
                time_manual.maxDurationReached = True
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=time_manual,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                time_manual.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if time_manual.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in time_manual.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "time_manual" ---
        for thisComponent in time_manual.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for time_manual
        time_manual.tStop = globalClock.getTime(format='float')
        time_manual.tStopRefresh = tThisFlipGlobal
        thisExp.addData('time_manual.stopped', time_manual.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if time_manual.maxDurationReached:
            routineTimer.addTime(-time_manual.maxDuration)
        elif time_manual.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fixation_cross_look],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_START | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisManual, 'status') and thisManual.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_look* updates
            
            # if fixation_cross_look is starting this frame...
            if fixation_cross_look.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_look.frameNStart = frameN  # exact frame index
                fixation_cross_look.tStart = t  # local t and not account for scr refresh
                fixation_cross_look.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_look, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_look.started')
                # update status
                fixation_cross_look.status = STARTED
                fixation_cross_look.setAutoDraw(True)
            
            # if fixation_cross_look is active this frame...
            if fixation_cross_look.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_look is stopping this frame...
            if fixation_cross_look.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_look.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_look.tStop = t  # not accounting for scr refresh
                    fixation_cross_look.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross_look.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_look.stopped')
                    # update status
                    fixation_cross_look.status = FINISHED
                    fixation_cross_look.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # Run 'End Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_END | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_manual" ---
        # create an object to store info about Routine trial_manual
        trial_manual = data.Routine(
            name='trial_manual',
            components=[trial_manual_aircraft_1, trial_manual_aircraft_2, change_manual, maintain_manual],
        )
        trial_manual.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_manual_aircraft_1.setText(a1)
        trial_manual_aircraft_2.setText(a2)
        # reset change_manual to account for continued clicks & clear times on/off
        change_manual.reset()
        # reset maintain_manual to account for continued clicks & clear times on/off
        maintain_manual.reset()
        # Run 'Begin Routine' code from code_manual
        # Start time
        t_start = globalClock.getTime()
        
        # Marker with trial information
        marker_info_manual = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Aircraft_1={a1} | "
            f"Aircraft_2={a2} | "
            f"Loss_of_separation={conflict}"
        )
        outlet.push_sample([marker_info_manual])
        
        # Trial start marker
        marker_start_manual = f"START | T{block_trial} | B{block}"
        outlet.push_sample([marker_start_manual])
        # store start times for trial_manual
        trial_manual.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_manual.tStart = globalClock.getTime(format='float')
        trial_manual.status = STARTED
        thisExp.addData('trial_manual.started', trial_manual.tStart)
        trial_manual.maxDuration = 3
        # keep track of which components have finished
        trial_manualComponents = trial_manual.components
        for thisComponent in trial_manual.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_manual" ---
        thisExp.currentRoutine = trial_manual
        trial_manual.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisManual, 'status') and thisManual.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > trial_manual.maxDuration-frameTolerance:
                trial_manual.maxDurationReached = True
                continueRoutine = False
            
            # *trial_manual_aircraft_1* updates
            
            # if trial_manual_aircraft_1 is starting this frame...
            if trial_manual_aircraft_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_manual_aircraft_1.frameNStart = frameN  # exact frame index
                trial_manual_aircraft_1.tStart = t  # local t and not account for scr refresh
                trial_manual_aircraft_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_manual_aircraft_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_manual_aircraft_1.started')
                # update status
                trial_manual_aircraft_1.status = STARTED
                trial_manual_aircraft_1.setAutoDraw(True)
            
            # if trial_manual_aircraft_1 is active this frame...
            if trial_manual_aircraft_1.status == STARTED:
                # update params
                pass
            
            # *trial_manual_aircraft_2* updates
            
            # if trial_manual_aircraft_2 is starting this frame...
            if trial_manual_aircraft_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                trial_manual_aircraft_2.frameNStart = frameN  # exact frame index
                trial_manual_aircraft_2.tStart = t  # local t and not account for scr refresh
                trial_manual_aircraft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_manual_aircraft_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_manual_aircraft_2.status = STARTED
                trial_manual_aircraft_2.setAutoDraw(True)
            
            # if trial_manual_aircraft_2 is active this frame...
            if trial_manual_aircraft_2.status == STARTED:
                # update params
                pass
            # *change_manual* updates
            
            # if change_manual is starting this frame...
            if change_manual.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                change_manual.frameNStart = frameN  # exact frame index
                change_manual.tStart = t  # local t and not account for scr refresh
                change_manual.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(change_manual, 'tStartRefresh')  # time at next scr refresh
                # update status
                change_manual.status = STARTED
                win.callOnFlip(change_manual.buttonClock.reset)
                change_manual.setAutoDraw(True)
            
            # if change_manual is active this frame...
            if change_manual.status == STARTED:
                # update params
                pass
                # check whether change_manual has been pressed
                if change_manual.isClicked:
                    if not change_manual.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        change_manual.timesOn.append(change_manual.buttonClock.getTime())
                        change_manual.timesOff.append(change_manual.buttonClock.getTime())
                    elif len(change_manual.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        change_manual.timesOff[-1] = change_manual.buttonClock.getTime()
                    if not change_manual.wasClicked:
                        # end routine when change_manual is clicked
                        continueRoutine = False
                    if not change_manual.wasClicked:
                        # run callback code when change_manual is clicked
                        pass
            # take note of whether change_manual was clicked, so that next frame we know if clicks are new
            change_manual.wasClicked = change_manual.isClicked and change_manual.status == STARTED
            # *maintain_manual* updates
            
            # if maintain_manual is starting this frame...
            if maintain_manual.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                maintain_manual.frameNStart = frameN  # exact frame index
                maintain_manual.tStart = t  # local t and not account for scr refresh
                maintain_manual.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maintain_manual, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maintain_manual.started')
                # update status
                maintain_manual.status = STARTED
                win.callOnFlip(maintain_manual.buttonClock.reset)
                maintain_manual.setAutoDraw(True)
            
            # if maintain_manual is active this frame...
            if maintain_manual.status == STARTED:
                # update params
                pass
                # check whether maintain_manual has been pressed
                if maintain_manual.isClicked:
                    if not maintain_manual.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        maintain_manual.timesOn.append(maintain_manual.buttonClock.getTime())
                        maintain_manual.timesOff.append(maintain_manual.buttonClock.getTime())
                    elif len(maintain_manual.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        maintain_manual.timesOff[-1] = maintain_manual.buttonClock.getTime()
                    if not maintain_manual.wasClicked:
                        # end routine when maintain_manual is clicked
                        continueRoutine = False
                    if not maintain_manual.wasClicked:
                        # run callback code when maintain_manual is clicked
                        pass
            # take note of whether maintain_manual was clicked, so that next frame we know if clicks are new
            maintain_manual.wasClicked = maintain_manual.isClicked and maintain_manual.status == STARTED
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_manual,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_manual.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_manual.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_manual.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_manual" ---
        for thisComponent in trial_manual.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_manual
        trial_manual.tStop = globalClock.getTime(format='float')
        trial_manual.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_manual.stopped', trial_manual.tStop)
        manual.addData('change_manual.numClicks', change_manual.numClicks)
        if change_manual.numClicks:
           manual.addData('change_manual.timesOn', change_manual.timesOn)
           manual.addData('change_manual.timesOff', change_manual.timesOff)
        else:
           manual.addData('change_manual.timesOn', "")
           manual.addData('change_manual.timesOff', "")
        manual.addData('maintain_manual.numClicks', maintain_manual.numClicks)
        if maintain_manual.numClicks:
           manual.addData('maintain_manual.timesOn', maintain_manual.timesOn)
           manual.addData('maintain_manual.timesOff', maintain_manual.timesOff)
        else:
           manual.addData('maintain_manual.timesOn', "")
           manual.addData('maintain_manual.timesOff', "")
        # Run 'End Routine' code from code_manual
        # Detect which button was pressed
        if change_manual.isClicked:
            response = "change"
        elif maintain_manual.isClicked:
            response = "maintain"
        else:
            response = "no_response"
        
        # Normalize Excel value
        con = str(conflict).strip().lower()
        
        # Evaluate correctness
        if con == "yes" and response == "change":
            result = "Correct action"
            correct = 1
        elif con == "no" and response == "maintain":
            result = "Correct action"
            correct = 1
        elif con == "yes" and response == "maintain":
            result = "Incorrect action"
            correct = 0
        elif con == "no" and response == "change":
            result = "Incorrect action"
            correct = 0
        else:
            result = "Time out"
            correct = 0
        
        # Save data
        thisExp.addData('answer', response)
        thisExp.addData('correct', correct)
        
        # Marker with trial information
        marker_info_manual = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Response={response} | "
            f"Correct={correct}"
        )
        outlet.push_sample([marker_info_manual])
        
        # Trial end marker
        marker_end_manual = f"END | T{block_trial} | B{block}"
        outlet.push_sample([marker_end_manual])
        # the Routine "trial_manual" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "answer_manual" ---
        # create an object to store info about Routine answer_manual
        answer_manual = data.Routine(
            name='answer_manual',
            components=[text_answer_manual],
        )
        answer_manual.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_answer_manual.setText(result)
        # store start times for answer_manual
        answer_manual.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        answer_manual.tStart = globalClock.getTime(format='float')
        answer_manual.status = STARTED
        thisExp.addData('answer_manual.started', answer_manual.tStart)
        answer_manual.maxDuration = None
        # keep track of which components have finished
        answer_manualComponents = answer_manual.components
        for thisComponent in answer_manual.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "answer_manual" ---
        thisExp.currentRoutine = answer_manual
        answer_manual.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisManual, 'status') and thisManual.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_answer_manual* updates
            
            # if text_answer_manual is starting this frame...
            if text_answer_manual.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer_manual.frameNStart = frameN  # exact frame index
                text_answer_manual.tStart = t  # local t and not account for scr refresh
                text_answer_manual.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer_manual, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_manual.started')
                # update status
                text_answer_manual.status = STARTED
                text_answer_manual.setAutoDraw(True)
            
            # if text_answer_manual is active this frame...
            if text_answer_manual.status == STARTED:
                # update params
                pass
            
            # if text_answer_manual is stopping this frame...
            if text_answer_manual.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_answer_manual.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_answer_manual.tStop = t  # not accounting for scr refresh
                    text_answer_manual.tStopRefresh = tThisFlipGlobal  # on global time
                    text_answer_manual.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_answer_manual.stopped')
                    # update status
                    text_answer_manual.status = FINISHED
                    text_answer_manual.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=answer_manual,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                answer_manual.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if answer_manual.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in answer_manual.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "answer_manual" ---
        for thisComponent in answer_manual.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for answer_manual
        answer_manual.tStop = globalClock.getTime(format='float')
        answer_manual.tStopRefresh = tThisFlipGlobal
        thisExp.addData('answer_manual.stopped', answer_manual.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if answer_manual.maxDurationReached:
            routineTimer.addTime(-answer_manual.maxDuration)
        elif answer_manual.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        # mark thisManual as finished
        if hasattr(thisManual, 'status'):
            thisManual.status = FINISHED
        # if awaiting a pause, pause now
        if manual.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            manual.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'manual'
    manual.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "high_reliability_start" ---
    # create an object to store info about Routine high_reliability_start
    high_reliability_start = data.Routine(
        name='high_reliability_start',
        components=[text_high_reliability_start, key_resp_high_reliability_start, text_space_high_reliability_start],
    )
    high_reliability_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_high_reliability_start
    key_resp_high_reliability_start.keys = []
    key_resp_high_reliability_start.rt = []
    _key_resp_high_reliability_start_allKeys = []
    # store start times for high_reliability_start
    high_reliability_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    high_reliability_start.tStart = globalClock.getTime(format='float')
    high_reliability_start.status = STARTED
    thisExp.addData('high_reliability_start.started', high_reliability_start.tStart)
    high_reliability_start.maxDuration = None
    # keep track of which components have finished
    high_reliability_startComponents = high_reliability_start.components
    for thisComponent in high_reliability_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "high_reliability_start" ---
    thisExp.currentRoutine = high_reliability_start
    high_reliability_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_high_reliability_start* updates
        
        # if text_high_reliability_start is starting this frame...
        if text_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_high_reliability_start.frameNStart = frameN  # exact frame index
            text_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_high_reliability_start.status = STARTED
            text_high_reliability_start.setAutoDraw(True)
        
        # if text_high_reliability_start is active this frame...
        if text_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # *key_resp_high_reliability_start* updates
        
        # if key_resp_high_reliability_start is starting this frame...
        if key_resp_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_high_reliability_start.frameNStart = frameN  # exact frame index
            key_resp_high_reliability_start.tStart = t  # local t and not account for scr refresh
            key_resp_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_high_reliability_start.status = STARTED
            # keyboard checking is just starting
            key_resp_high_reliability_start.clock.reset()  # now t=0
        if key_resp_high_reliability_start.status == STARTED:
            theseKeys = key_resp_high_reliability_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_high_reliability_start_allKeys.extend(theseKeys)
            if len(_key_resp_high_reliability_start_allKeys):
                key_resp_high_reliability_start.keys = _key_resp_high_reliability_start_allKeys[-1].name  # just the last key pressed
                key_resp_high_reliability_start.rt = _key_resp_high_reliability_start_allKeys[-1].rt
                key_resp_high_reliability_start.duration = _key_resp_high_reliability_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_high_reliability_start* updates
        
        # if text_space_high_reliability_start is starting this frame...
        if text_space_high_reliability_start.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_high_reliability_start.frameNStart = frameN  # exact frame index
            text_space_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_space_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_high_reliability_start.status = STARTED
            text_space_high_reliability_start.setAutoDraw(True)
        
        # if text_space_high_reliability_start is active this frame...
        if text_space_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=high_reliability_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            high_reliability_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if high_reliability_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in high_reliability_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "high_reliability_start" ---
    for thisComponent in high_reliability_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for high_reliability_start
    high_reliability_start.tStop = globalClock.getTime(format='float')
    high_reliability_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('high_reliability_start.stopped', high_reliability_start.tStop)
    thisExp.nextEntry()
    # the Routine "high_reliability_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    high_1 = data.TrialHandler2(
        name='high_1',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trial_100_1_ecological.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(high_1)  # add the loop to the experiment
    thisHigh_1 = high_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisHigh_1.rgb)
    if thisHigh_1 != None:
        for paramName in thisHigh_1:
            globals()[paramName] = thisHigh_1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisHigh_1 in high_1:
        high_1.status = STARTED
        if hasattr(thisHigh_1, 'status'):
            thisHigh_1.status = STARTED
        currentLoop = high_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisHigh_1.rgb)
        if thisHigh_1 != None:
            for paramName in thisHigh_1:
                globals()[paramName] = thisHigh_1[paramName]
        
        # --- Prepare to start Routine "time_high" ---
        # create an object to store info about Routine time_high
        time_high = data.Routine(
            name='time_high',
            components=[],
        )
        time_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for time_high
        time_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_high.tStart = globalClock.getTime(format='float')
        time_high.status = STARTED
        time_high.maxDuration = 1
        # keep track of which components have finished
        time_highComponents = time_high.components
        for thisComponent in time_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "time_high" ---
        thisExp.currentRoutine = time_high
        time_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_1, 'status') and thisHigh_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > time_high.maxDuration-frameTolerance:
                time_high.maxDurationReached = True
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=time_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                time_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if time_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in time_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "time_high" ---
        for thisComponent in time_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for time_high
        time_high.tStop = globalClock.getTime(format='float')
        time_high.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if time_high.maxDurationReached:
            routineTimer.addTime(-time_high.maxDuration)
        elif time_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fixation_cross_look],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_START | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_1, 'status') and thisHigh_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_look* updates
            
            # if fixation_cross_look is starting this frame...
            if fixation_cross_look.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_look.frameNStart = frameN  # exact frame index
                fixation_cross_look.tStart = t  # local t and not account for scr refresh
                fixation_cross_look.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_look, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_look.started')
                # update status
                fixation_cross_look.status = STARTED
                fixation_cross_look.setAutoDraw(True)
            
            # if fixation_cross_look is active this frame...
            if fixation_cross_look.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_look is stopping this frame...
            if fixation_cross_look.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_look.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_look.tStop = t  # not accounting for scr refresh
                    fixation_cross_look.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross_look.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_look.stopped')
                    # update status
                    fixation_cross_look.status = FINISHED
                    fixation_cross_look.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # Run 'End Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_END | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_high" ---
        # create an object to store info about Routine trial_high
        trial_high = data.Routine(
            name='trial_high',
            components=[trial_high_aircraft_1, trial_high_aircraft_2, change_high, maintain_high, support_high],
        )
        trial_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_high_aircraft_1.setText(a1)
        trial_high_aircraft_2.setText(a2)
        # reset change_high to account for continued clicks & clear times on/off
        change_high.reset()
        # reset maintain_high to account for continued clicks & clear times on/off
        maintain_high.reset()
        support_high.setPos((value, -0.3))
        # Run 'Begin Routine' code from code_high
        # Start time
        t_start = globalClock.getTime()
        
        # Interpret automated recommendation
        if value < 0:
            recommendation = "change"
        else:
            recommendation = "maintain"
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Aircraft_1={a1} | "
            f"Aircraft_2={a2} | "
            f"Loss_of_separation={conflict} | "
            f"Automation_recommendation={recommendation} | "
            f"Automation_error={automation_error}"
        )
        outlet.push_sample([marker_info_low])
        
        # Start marker
        marker_start_low = f"START | T{block_trial} | B{block}"
        outlet.push_sample([marker_start_low])
        # store start times for trial_high
        trial_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_high.tStart = globalClock.getTime(format='float')
        trial_high.status = STARTED
        thisExp.addData('trial_high.started', trial_high.tStart)
        trial_high.maxDuration = 3
        # keep track of which components have finished
        trial_highComponents = trial_high.components
        for thisComponent in trial_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_high" ---
        thisExp.currentRoutine = trial_high
        trial_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_1, 'status') and thisHigh_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > trial_high.maxDuration-frameTolerance:
                trial_high.maxDurationReached = True
                continueRoutine = False
            
            # *trial_high_aircraft_1* updates
            
            # if trial_high_aircraft_1 is starting this frame...
            if trial_high_aircraft_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_1.frameNStart = frameN  # exact frame index
                trial_high_aircraft_1.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_high_aircraft_1.started')
                # update status
                trial_high_aircraft_1.status = STARTED
                trial_high_aircraft_1.setAutoDraw(True)
            
            # if trial_high_aircraft_1 is active this frame...
            if trial_high_aircraft_1.status == STARTED:
                # update params
                pass
            
            # *trial_high_aircraft_2* updates
            
            # if trial_high_aircraft_2 is starting this frame...
            if trial_high_aircraft_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_2.frameNStart = frameN  # exact frame index
                trial_high_aircraft_2.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_high_aircraft_2.status = STARTED
                trial_high_aircraft_2.setAutoDraw(True)
            
            # if trial_high_aircraft_2 is active this frame...
            if trial_high_aircraft_2.status == STARTED:
                # update params
                pass
            # *change_high* updates
            
            # if change_high is starting this frame...
            if change_high.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                change_high.frameNStart = frameN  # exact frame index
                change_high.tStart = t  # local t and not account for scr refresh
                change_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(change_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                change_high.status = STARTED
                win.callOnFlip(change_high.buttonClock.reset)
                change_high.setAutoDraw(True)
            
            # if change_high is active this frame...
            if change_high.status == STARTED:
                # update params
                pass
                # check whether change_high has been pressed
                if change_high.isClicked:
                    if not change_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        change_high.timesOn.append(change_high.buttonClock.getTime())
                        change_high.timesOff.append(change_high.buttonClock.getTime())
                    elif len(change_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        change_high.timesOff[-1] = change_high.buttonClock.getTime()
                    if not change_high.wasClicked:
                        # end routine when change_high is clicked
                        continueRoutine = False
                    if not change_high.wasClicked:
                        # run callback code when change_high is clicked
                        pass
            # take note of whether change_high was clicked, so that next frame we know if clicks are new
            change_high.wasClicked = change_high.isClicked and change_high.status == STARTED
            # *maintain_high* updates
            
            # if maintain_high is starting this frame...
            if maintain_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                maintain_high.frameNStart = frameN  # exact frame index
                maintain_high.tStart = t  # local t and not account for scr refresh
                maintain_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maintain_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maintain_high.started')
                # update status
                maintain_high.status = STARTED
                win.callOnFlip(maintain_high.buttonClock.reset)
                maintain_high.setAutoDraw(True)
            
            # if maintain_high is active this frame...
            if maintain_high.status == STARTED:
                # update params
                pass
                # check whether maintain_high has been pressed
                if maintain_high.isClicked:
                    if not maintain_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        maintain_high.timesOn.append(maintain_high.buttonClock.getTime())
                        maintain_high.timesOff.append(maintain_high.buttonClock.getTime())
                    elif len(maintain_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        maintain_high.timesOff[-1] = maintain_high.buttonClock.getTime()
                    if not maintain_high.wasClicked:
                        # end routine when maintain_high is clicked
                        continueRoutine = False
                    if not maintain_high.wasClicked:
                        # run callback code when maintain_high is clicked
                        pass
            # take note of whether maintain_high was clicked, so that next frame we know if clicks are new
            maintain_high.wasClicked = maintain_high.isClicked and maintain_high.status == STARTED
            
            # *support_high* updates
            
            # if support_high is starting this frame...
            if support_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                support_high.frameNStart = frameN  # exact frame index
                support_high.tStart = t  # local t and not account for scr refresh
                support_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(support_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'support_high.started')
                # update status
                support_high.status = STARTED
                support_high.setAutoDraw(True)
            
            # if support_high is active this frame...
            if support_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_high" ---
        for thisComponent in trial_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_high
        trial_high.tStop = globalClock.getTime(format='float')
        trial_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_high.stopped', trial_high.tStop)
        high_1.addData('change_high.numClicks', change_high.numClicks)
        if change_high.numClicks:
           high_1.addData('change_high.timesOn', change_high.timesOn)
           high_1.addData('change_high.timesOff', change_high.timesOff)
        else:
           high_1.addData('change_high.timesOn', "")
           high_1.addData('change_high.timesOff', "")
        high_1.addData('maintain_high.numClicks', maintain_high.numClicks)
        if maintain_high.numClicks:
           high_1.addData('maintain_high.timesOn', maintain_high.timesOn)
           high_1.addData('maintain_high.timesOff', maintain_high.timesOff)
        else:
           high_1.addData('maintain_high.timesOn', "")
           high_1.addData('maintain_high.timesOff', "")
        # Run 'End Routine' code from code_high
        # Detect which button was pressed
        if change_high.isClicked:
            response = "change"
        elif maintain_high.isClicked:
            response = "maintain"
        else:
            response = "no_response"
        
        # Normalize Excel value
        con = str(conflict).strip().lower()
        
        # Evaluate correctness
        if con == "yes" and response == "change":
            result = "Correct action"
            correct = 1
        elif con == "no" and response == "maintain":
            result = "Correct action"
            correct = 1
        elif con == "yes" and response == "maintain":
            result = "Incorrect action"
            correct = 0
        elif con == "no" and response == "change":
            result = "Incorrect action"
            correct = 0
        else:
            result = "Time out"
            correct = 0
        
        # Save data
        thisExp.addData('answer', response)
        thisExp.addData('correct', correct)
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Response={response} | "
            f"Correct={correct}"
        )
        outlet.push_sample([marker_info_low])
        
        # Trial end marker
        marker_end_low = f"END | T{block_trial} | B{block}"
        outlet.push_sample([marker_end_low])
        # the Routine "trial_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "answer_high" ---
        # create an object to store info about Routine answer_high
        answer_high = data.Routine(
            name='answer_high',
            components=[text_answer_high],
        )
        answer_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_answer_high.setText(result)
        # store start times for answer_high
        answer_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        answer_high.tStart = globalClock.getTime(format='float')
        answer_high.status = STARTED
        thisExp.addData('answer_high.started', answer_high.tStart)
        answer_high.maxDuration = None
        # keep track of which components have finished
        answer_highComponents = answer_high.components
        for thisComponent in answer_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "answer_high" ---
        thisExp.currentRoutine = answer_high
        answer_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_1, 'status') and thisHigh_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_answer_high* updates
            
            # if text_answer_high is starting this frame...
            if text_answer_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer_high.frameNStart = frameN  # exact frame index
                text_answer_high.tStart = t  # local t and not account for scr refresh
                text_answer_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_high.started')
                # update status
                text_answer_high.status = STARTED
                text_answer_high.setAutoDraw(True)
            
            # if text_answer_high is active this frame...
            if text_answer_high.status == STARTED:
                # update params
                pass
            
            # if text_answer_high is stopping this frame...
            if text_answer_high.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_answer_high.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_answer_high.tStop = t  # not accounting for scr refresh
                    text_answer_high.tStopRefresh = tThisFlipGlobal  # on global time
                    text_answer_high.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_answer_high.stopped')
                    # update status
                    text_answer_high.status = FINISHED
                    text_answer_high.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=answer_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                answer_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if answer_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in answer_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "answer_high" ---
        for thisComponent in answer_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for answer_high
        answer_high.tStop = globalClock.getTime(format='float')
        answer_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('answer_high.stopped', answer_high.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if answer_high.maxDurationReached:
            routineTimer.addTime(-answer_high.maxDuration)
        elif answer_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_high" ---
        # create an object to store info about Routine question_high
        question_high = data.Routine(
            name='question_high',
            components=[text_high_question, slider_high_question, mouse_high_question],
        )
        question_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_high_question.setText('How much do you trust the automation?')
        slider_high_question.reset()
        # setup some python lists for storing info about the mouse_high_question
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_high_question
        # Reset mouse clicks
        mouse_high_question.clickReset()
        
        # Previous button state
        prevButtonState = mouse_high_question.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        # store start times for question_high
        question_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        question_high.tStart = globalClock.getTime(format='float')
        question_high.status = STARTED
        thisExp.addData('question_high.started', question_high.tStart)
        question_high.maxDuration = None
        # keep track of which components have finished
        question_highComponents = question_high.components
        for thisComponent in question_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_high" ---
        thisExp.currentRoutine = question_high
        question_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_1, 'status') and thisHigh_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_high_question* updates
            
            # if text_high_question is starting this frame...
            if text_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_high_question.frameNStart = frameN  # exact frame index
                text_high_question.tStart = t  # local t and not account for scr refresh
                text_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_high_question.status = STARTED
                text_high_question.setAutoDraw(True)
            
            # if text_high_question is active this frame...
            if text_high_question.status == STARTED:
                # update params
                pass
            
            # *slider_high_question* updates
            
            # if slider_high_question is starting this frame...
            if slider_high_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_high_question.frameNStart = frameN  # exact frame index
                slider_high_question.tStart = t  # local t and not account for scr refresh
                slider_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_high_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_high_question.started')
                # update status
                slider_high_question.status = STARTED
                slider_high_question.setAutoDraw(True)
            
            # if slider_high_question is active this frame...
            if slider_high_question.status == STARTED:
                # update params
                pass
            
            # Check slider_high_question for response to end Routine
            if slider_high_question.getRating() is not None and slider_high_question.status == STARTED:
                continueRoutine = False
            # *mouse_high_question* updates
            
            # if mouse_high_question is starting this frame...
            if mouse_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_high_question.frameNStart = frameN  # exact frame index
                mouse_high_question.tStart = t  # local t and not account for scr refresh
                mouse_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_high_question.status = STARTED
                mouse_high_question.mouseClock.reset()
                prevButtonState = mouse_high_question.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_high_question
            mouse_high_question_pos = mouse_high_question.getPos()
            
            x_min = slider_high_question.pos[0] - slider_high_question.size[0] / 2
            x_max = slider_high_question.pos[0] + slider_high_question.size[0] / 2
            y_min = slider_high_question.pos[1] - 0.1
            y_max = slider_high_question.pos[1] + 0.1
            
            # Move marker when hovering over the slider
            if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                slider_high_question.markerPos = slider_high_question._posToRating(mouse_high_question_pos)
            
            buttonState = mouse_high_question.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                    slider_high_question.rating = slider_high_question.markerPos
                    continueRoutine = False
            
            prevButtonState = buttonState
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=question_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                question_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if question_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in question_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_high" ---
        for thisComponent in question_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for question_high
        question_high.tStop = globalClock.getTime(format='float')
        question_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('question_high.stopped', question_high.tStop)
        # store data for high_1 (TrialHandler)
        # Run 'End Routine' code from code_high_question
        thisExp.addData('trial', block_trial)
        thisExp.addData('answ_high_question', slider_high_question.markerPos)
        
        if slider_high_question.markerPos is not None:
            trust_answer = slider_high_question.markerPos
        else:
            trust_answer = "NA"
        
        marker = f"CONF_H | B{block} | T{block_trial} | V={trust_answer}"
        outlet.push_sample([marker])
        # the Routine "question_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisHigh_1 as finished
        if hasattr(thisHigh_1, 'status'):
            thisHigh_1.status = FINISHED
        # if awaiting a pause, pause now
        if high_1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            high_1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'high_1'
    high_1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "SATI_short_high_instructions" ---
    # create an object to store info about Routine SATI_short_high_instructions
    SATI_short_high_instructions = data.Routine(
        name='SATI_short_high_instructions',
        components=[text_SATI_short_high_instructions, key_resp_SATI_short_high_instructions, text_space_SATI_short_high_instructions],
    )
    SATI_short_high_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_SATI_short_high_instructions
    key_resp_SATI_short_high_instructions.keys = []
    key_resp_SATI_short_high_instructions.rt = []
    _key_resp_SATI_short_high_instructions_allKeys = []
    # store start times for SATI_short_high_instructions
    SATI_short_high_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SATI_short_high_instructions.tStart = globalClock.getTime(format='float')
    SATI_short_high_instructions.status = STARTED
    thisExp.addData('SATI_short_high_instructions.started', SATI_short_high_instructions.tStart)
    SATI_short_high_instructions.maxDuration = None
    # keep track of which components have finished
    SATI_short_high_instructionsComponents = SATI_short_high_instructions.components
    for thisComponent in SATI_short_high_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SATI_short_high_instructions" ---
    thisExp.currentRoutine = SATI_short_high_instructions
    SATI_short_high_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_SATI_short_high_instructions* updates
        
        # if text_SATI_short_high_instructions is starting this frame...
        if text_SATI_short_high_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_SATI_short_high_instructions.started')
            # update status
            text_SATI_short_high_instructions.status = STARTED
            text_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_SATI_short_high_instructions is active this frame...
        if text_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_SATI_short_high_instructions* updates
        
        # if key_resp_SATI_short_high_instructions is starting this frame...
        if key_resp_SATI_short_high_instructions.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            key_resp_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_SATI_short_high_instructions.status = STARTED
            # keyboard checking is just starting
            key_resp_SATI_short_high_instructions.clock.reset()  # now t=0
        if key_resp_SATI_short_high_instructions.status == STARTED:
            theseKeys = key_resp_SATI_short_high_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_SATI_short_high_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_SATI_short_high_instructions_allKeys):
                key_resp_SATI_short_high_instructions.keys = _key_resp_SATI_short_high_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_SATI_short_high_instructions.rt = _key_resp_SATI_short_high_instructions_allKeys[-1].rt
                key_resp_SATI_short_high_instructions.duration = _key_resp_SATI_short_high_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_SATI_short_high_instructions* updates
        
        # if text_space_SATI_short_high_instructions is starting this frame...
        if text_space_SATI_short_high_instructions.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_space_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_space_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_SATI_short_high_instructions.status = STARTED
            text_space_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_space_SATI_short_high_instructions is active this frame...
        if text_space_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SATI_short_high_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            SATI_short_high_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if SATI_short_high_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in SATI_short_high_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SATI_short_high_instructions" ---
    for thisComponent in SATI_short_high_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SATI_short_high_instructions
    SATI_short_high_instructions.tStop = globalClock.getTime(format='float')
    SATI_short_high_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SATI_short_high_instructions.stopped', SATI_short_high_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "SATI_short_high_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    SATI_high_1 = data.TrialHandler2(
        name='SATI_high_1',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('SATI.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(SATI_high_1)  # add the loop to the experiment
    thisSATI_high_1 = SATI_high_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSATI_high_1.rgb)
    if thisSATI_high_1 != None:
        for paramName in thisSATI_high_1:
            globals()[paramName] = thisSATI_high_1[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSATI_high_1 in SATI_high_1:
        SATI_high_1.status = STARTED
        if hasattr(thisSATI_high_1, 'status'):
            thisSATI_high_1.status = STARTED
        currentLoop = SATI_high_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSATI_high_1.rgb)
        if thisSATI_high_1 != None:
            for paramName in thisSATI_high_1:
                globals()[paramName] = thisSATI_high_1[paramName]
        
        # --- Prepare to start Routine "SATI_short_high" ---
        # create an object to store info about Routine SATI_short_high
        SATI_short_high = data.Routine(
            name='SATI_short_high',
            components=[text_SATI_short_high, text_SATI_short_high_clarification, slider_SATI_short_high, mouse_SATI_short_high, prog_short_high],
        )
        SATI_short_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_SATI_short_high.setText(Question)
        text_SATI_short_high_clarification.setText(Clarification)
        slider_SATI_short_high.reset()
        # setup some python lists for storing info about the mouse_SATI_short_high
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_SATI_short_high
        # Reset mouse clicks
        mouse_SATI_short_high.clickReset()
        
        # Previous button state (important for detecting a clean click)
        prevButtonState = mouse_SATI_short_high.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_short_high.setProgress((block_trial - 1) / 6)
        # store start times for SATI_short_high
        SATI_short_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SATI_short_high.tStart = globalClock.getTime(format='float')
        SATI_short_high.status = STARTED
        thisExp.addData('SATI_short_high.started', SATI_short_high.tStart)
        SATI_short_high.maxDuration = None
        # keep track of which components have finished
        SATI_short_highComponents = SATI_short_high.components
        for thisComponent in SATI_short_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SATI_short_high" ---
        thisExp.currentRoutine = SATI_short_high
        SATI_short_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSATI_high_1, 'status') and thisSATI_high_1.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_SATI_short_high* updates
            
            # if text_SATI_short_high is starting this frame...
            if text_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high.frameNStart = frameN  # exact frame index
                text_SATI_short_high.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high.status = STARTED
                text_SATI_short_high.setAutoDraw(True)
            
            # if text_SATI_short_high is active this frame...
            if text_SATI_short_high.status == STARTED:
                # update params
                pass
            
            # *text_SATI_short_high_clarification* updates
            
            # if text_SATI_short_high_clarification is starting this frame...
            if text_SATI_short_high_clarification.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high_clarification.frameNStart = frameN  # exact frame index
                text_SATI_short_high_clarification.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high_clarification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high_clarification, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high_clarification.status = STARTED
                text_SATI_short_high_clarification.setAutoDraw(True)
            
            # if text_SATI_short_high_clarification is active this frame...
            if text_SATI_short_high_clarification.status == STARTED:
                # update params
                pass
            
            # *slider_SATI_short_high* updates
            
            # if slider_SATI_short_high is starting this frame...
            if slider_SATI_short_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_SATI_short_high.frameNStart = frameN  # exact frame index
                slider_SATI_short_high.tStart = t  # local t and not account for scr refresh
                slider_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SATI_short_high.started')
                # update status
                slider_SATI_short_high.status = STARTED
                slider_SATI_short_high.setAutoDraw(True)
            
            # if slider_SATI_short_high is active this frame...
            if slider_SATI_short_high.status == STARTED:
                # update params
                pass
            # *mouse_SATI_short_high* updates
            
            # if mouse_SATI_short_high is starting this frame...
            if mouse_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_SATI_short_high.frameNStart = frameN  # exact frame index
                mouse_SATI_short_high.tStart = t  # local t and not account for scr refresh
                mouse_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_SATI_short_high.status = STARTED
                mouse_SATI_short_high.mouseClock.reset()
                prevButtonState = mouse_SATI_short_high.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_SATI_short_high
            mouse_pos = mouse_SATI_short_high.getPos()
            
            x_min = slider_SATI_short_high.pos[0] - slider_SATI_short_high.size[0] / 2
            x_max = slider_SATI_short_high.pos[0] + slider_SATI_short_high.size[0] / 2
            y_min = slider_SATI_short_high.pos[1] - 0.1
            y_max = slider_SATI_short_high.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_SATI_short_high.markerPos = slider_SATI_short_high._posToRating(mouse_pos)
            
            buttonState = mouse_SATI_short_high.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_SATI_short_high.rating = slider_SATI_short_high.markerPos
                    responseSelected = True
            
            # Wait until the mouse has been released for 0.8 s
            if responseSelected:
                if buttonState == 0:
                    if releaseStartTime is None:
                        releaseStartTime = t
                    elif t - releaseStartTime >= releaseWaitTime:
                        continueRoutine = False
                else:
                    releaseStartTime = None
            
            prevButtonState = buttonState
            
            # *prog_short_high* updates
            
            # if prog_short_high is starting this frame...
            if prog_short_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_short_high.frameNStart = frameN  # exact frame index
                prog_short_high.tStart = t  # local t and not account for scr refresh
                prog_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_short_high.started')
                # update status
                prog_short_high.status = STARTED
                prog_short_high.setAutoDraw(True)
            
            # if prog_short_high is active this frame...
            if prog_short_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=SATI_short_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                SATI_short_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if SATI_short_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in SATI_short_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SATI_short_high" ---
        for thisComponent in SATI_short_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SATI_short_high
        SATI_short_high.tStop = globalClock.getTime(format='float')
        SATI_short_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SATI_short_high.stopped', SATI_short_high.tStop)
        # store data for SATI_high_1 (TrialHandler)
        # Run 'End Routine' code from code_SATI_short_high
        if slider_SATI_short_high.markerPos is not None:
            thisExp.addData('answ_SATI_short_high', slider_SATI_short_high.markerPos)
        else:
            thisExp.addData('answ_SATI_short_high', 'NA')
        # the Routine "SATI_short_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisSATI_high_1 as finished
        if hasattr(thisSATI_high_1, 'status'):
            thisSATI_high_1.status = FINISHED
        # if awaiting a pause, pause now
        if SATI_high_1.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            SATI_high_1.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'SATI_high_1'
    SATI_high_1.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "how_you_feel_high" ---
    # create an object to store info about Routine how_you_feel_high
    how_you_feel_high = data.Routine(
        name='how_you_feel_high',
        components=[pleasant_high, unpleasant_high, text_how_you_feel_high],
    )
    how_you_feel_high.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset pleasant_high to account for continued clicks & clear times on/off
    pleasant_high.reset()
    # reset unpleasant_high to account for continued clicks & clear times on/off
    unpleasant_high.reset()
    # store start times for how_you_feel_high
    how_you_feel_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    how_you_feel_high.tStart = globalClock.getTime(format='float')
    how_you_feel_high.status = STARTED
    thisExp.addData('how_you_feel_high.started', how_you_feel_high.tStart)
    how_you_feel_high.maxDuration = None
    # keep track of which components have finished
    how_you_feel_highComponents = how_you_feel_high.components
    for thisComponent in how_you_feel_high.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "how_you_feel_high" ---
    thisExp.currentRoutine = how_you_feel_high
    how_you_feel_high.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pleasant_high* updates
        
        # if pleasant_high is starting this frame...
        if pleasant_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            pleasant_high.frameNStart = frameN  # exact frame index
            pleasant_high.tStart = t  # local t and not account for scr refresh
            pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            pleasant_high.status = STARTED
            win.callOnFlip(pleasant_high.buttonClock.reset)
            pleasant_high.setAutoDraw(True)
        
        # if pleasant_high is active this frame...
        if pleasant_high.status == STARTED:
            # update params
            pass
            # check whether pleasant_high has been pressed
            if pleasant_high.isClicked:
                if not pleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    pleasant_high.timesOn.append(pleasant_high.buttonClock.getTime())
                    pleasant_high.timesOff.append(pleasant_high.buttonClock.getTime())
                elif len(pleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    pleasant_high.timesOff[-1] = pleasant_high.buttonClock.getTime()
                if not pleasant_high.wasClicked:
                    # end routine when pleasant_high is clicked
                    continueRoutine = False
                if not pleasant_high.wasClicked:
                    # run callback code when pleasant_high is clicked
                    pass
        # take note of whether pleasant_high was clicked, so that next frame we know if clicks are new
        pleasant_high.wasClicked = pleasant_high.isClicked and pleasant_high.status == STARTED
        # *unpleasant_high* updates
        
        # if unpleasant_high is starting this frame...
        if unpleasant_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            unpleasant_high.frameNStart = frameN  # exact frame index
            unpleasant_high.tStart = t  # local t and not account for scr refresh
            unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'unpleasant_high.started')
            # update status
            unpleasant_high.status = STARTED
            win.callOnFlip(unpleasant_high.buttonClock.reset)
            unpleasant_high.setAutoDraw(True)
        
        # if unpleasant_high is active this frame...
        if unpleasant_high.status == STARTED:
            # update params
            pass
            # check whether unpleasant_high has been pressed
            if unpleasant_high.isClicked:
                if not unpleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    unpleasant_high.timesOn.append(unpleasant_high.buttonClock.getTime())
                    unpleasant_high.timesOff.append(unpleasant_high.buttonClock.getTime())
                elif len(unpleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    unpleasant_high.timesOff[-1] = unpleasant_high.buttonClock.getTime()
                if not unpleasant_high.wasClicked:
                    # end routine when unpleasant_high is clicked
                    continueRoutine = False
                if not unpleasant_high.wasClicked:
                    # run callback code when unpleasant_high is clicked
                    pass
        # take note of whether unpleasant_high was clicked, so that next frame we know if clicks are new
        unpleasant_high.wasClicked = unpleasant_high.isClicked and unpleasant_high.status == STARTED
        
        # *text_how_you_feel_high* updates
        
        # if text_how_you_feel_high is starting this frame...
        if text_how_you_feel_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_how_you_feel_high.frameNStart = frameN  # exact frame index
            text_how_you_feel_high.tStart = t  # local t and not account for scr refresh
            text_how_you_feel_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_how_you_feel_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_how_you_feel_high.started')
            # update status
            text_how_you_feel_high.status = STARTED
            text_how_you_feel_high.setAutoDraw(True)
        
        # if text_how_you_feel_high is active this frame...
        if text_how_you_feel_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=how_you_feel_high,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            how_you_feel_high.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if how_you_feel_high.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in how_you_feel_high.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "how_you_feel_high" ---
    for thisComponent in how_you_feel_high.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for how_you_feel_high
    how_you_feel_high.tStop = globalClock.getTime(format='float')
    how_you_feel_high.tStopRefresh = tThisFlipGlobal
    thisExp.addData('how_you_feel_high.stopped', how_you_feel_high.tStop)
    thisExp.addData('pleasant_high.numClicks', pleasant_high.numClicks)
    if pleasant_high.numClicks:
       thisExp.addData('pleasant_high.timesOn', pleasant_high.timesOn)
       thisExp.addData('pleasant_high.timesOff', pleasant_high.timesOff)
    else:
       thisExp.addData('pleasant_high.timesOn', "")
       thisExp.addData('pleasant_high.timesOff', "")
    thisExp.addData('unpleasant_high.numClicks', unpleasant_high.numClicks)
    if unpleasant_high.numClicks:
       thisExp.addData('unpleasant_high.timesOn', unpleasant_high.timesOn)
       thisExp.addData('unpleasant_high.timesOff', unpleasant_high.timesOff)
    else:
       thisExp.addData('unpleasant_high.timesOn', "")
       thisExp.addData('unpleasant_high.timesOff', "")
    # Run 'End Routine' code from code_how_you_feel_high
    if pleasant_high.isClicked:
        valence_answer = "pleasant"
    elif unpleasant_high.isClicked:
        valence_answer = "unpleasant"
    else:
        valence_answer = "no_response"
    
    thisExp.addData('answ_valence', valence_answer)
    thisExp.nextEntry()
    # the Routine "how_you_feel_high" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_high_choice" ---
    # create an object to store info about Routine pleasant_high_choice
    pleasant_high_choice = data.Routine(
        name='pleasant_high_choice',
        components=[excited_high, relaxed_high, text_pleasant_high],
    )
    pleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset excited_high to account for continued clicks & clear times on/off
    excited_high.reset()
    # reset relaxed_high to account for continued clicks & clear times on/off
    relaxed_high.reset()
    # Run 'Begin Routine' code from code_pleasant_high
    # Only show this routine if "pleasant" was selected
    if valence_answer != "pleasant":
        continueRoutine = False
    # store start times for pleasant_high_choice
    pleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pleasant_high_choice.tStart = globalClock.getTime(format='float')
    pleasant_high_choice.status = STARTED
    thisExp.addData('pleasant_high_choice.started', pleasant_high_choice.tStart)
    pleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    pleasant_high_choiceComponents = pleasant_high_choice.components
    for thisComponent in pleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_high_choice" ---
    thisExp.currentRoutine = pleasant_high_choice
    pleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *excited_high* updates
        
        # if excited_high is starting this frame...
        if excited_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            excited_high.frameNStart = frameN  # exact frame index
            excited_high.tStart = t  # local t and not account for scr refresh
            excited_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(excited_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            excited_high.status = STARTED
            win.callOnFlip(excited_high.buttonClock.reset)
            excited_high.setAutoDraw(True)
        
        # if excited_high is active this frame...
        if excited_high.status == STARTED:
            # update params
            pass
            # check whether excited_high has been pressed
            if excited_high.isClicked:
                if not excited_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    excited_high.timesOn.append(excited_high.buttonClock.getTime())
                    excited_high.timesOff.append(excited_high.buttonClock.getTime())
                elif len(excited_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    excited_high.timesOff[-1] = excited_high.buttonClock.getTime()
                if not excited_high.wasClicked:
                    # end routine when excited_high is clicked
                    continueRoutine = False
                if not excited_high.wasClicked:
                    # run callback code when excited_high is clicked
                    pass
        # take note of whether excited_high was clicked, so that next frame we know if clicks are new
        excited_high.wasClicked = excited_high.isClicked and excited_high.status == STARTED
        # *relaxed_high* updates
        
        # if relaxed_high is starting this frame...
        if relaxed_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            relaxed_high.frameNStart = frameN  # exact frame index
            relaxed_high.tStart = t  # local t and not account for scr refresh
            relaxed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(relaxed_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'relaxed_high.started')
            # update status
            relaxed_high.status = STARTED
            win.callOnFlip(relaxed_high.buttonClock.reset)
            relaxed_high.setAutoDraw(True)
        
        # if relaxed_high is active this frame...
        if relaxed_high.status == STARTED:
            # update params
            pass
            # check whether relaxed_high has been pressed
            if relaxed_high.isClicked:
                if not relaxed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    relaxed_high.timesOn.append(relaxed_high.buttonClock.getTime())
                    relaxed_high.timesOff.append(relaxed_high.buttonClock.getTime())
                elif len(relaxed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    relaxed_high.timesOff[-1] = relaxed_high.buttonClock.getTime()
                if not relaxed_high.wasClicked:
                    # end routine when relaxed_high is clicked
                    continueRoutine = False
                if not relaxed_high.wasClicked:
                    # run callback code when relaxed_high is clicked
                    pass
        # take note of whether relaxed_high was clicked, so that next frame we know if clicks are new
        relaxed_high.wasClicked = relaxed_high.isClicked and relaxed_high.status == STARTED
        
        # *text_pleasant_high* updates
        
        # if text_pleasant_high is starting this frame...
        if text_pleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pleasant_high.frameNStart = frameN  # exact frame index
            text_pleasant_high.tStart = t  # local t and not account for scr refresh
            text_pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pleasant_high.started')
            # update status
            text_pleasant_high.status = STARTED
            text_pleasant_high.setAutoDraw(True)
        
        # if text_pleasant_high is active this frame...
        if text_pleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=pleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            pleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if pleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in pleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_high_choice" ---
    for thisComponent in pleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pleasant_high_choice
    pleasant_high_choice.tStop = globalClock.getTime(format='float')
    pleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pleasant_high_choice.stopped', pleasant_high_choice.tStop)
    thisExp.addData('excited_high.numClicks', excited_high.numClicks)
    if excited_high.numClicks:
       thisExp.addData('excited_high.timesOn', excited_high.timesOn)
       thisExp.addData('excited_high.timesOff', excited_high.timesOff)
    else:
       thisExp.addData('excited_high.timesOn', "")
       thisExp.addData('excited_high.timesOff', "")
    thisExp.addData('relaxed_high.numClicks', relaxed_high.numClicks)
    if relaxed_high.numClicks:
       thisExp.addData('relaxed_high.timesOn', relaxed_high.timesOn)
       thisExp.addData('relaxed_high.timesOff', relaxed_high.timesOff)
    else:
       thisExp.addData('relaxed_high.timesOn', "")
       thisExp.addData('relaxed_high.timesOff', "")
    # Run 'End Routine' code from code_pleasant_high
    if excited_high.isClicked:
        pleasant_answer = "excited"
    elif relaxed_high.isClicked:
        pleasant_answer = "relaxed"
    else:
        pleasant_answer = "no_response"
    
    thisExp.addData('answ_pleasant', pleasant_answer)
    thisExp.nextEntry()
    # the Routine "pleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "unpleasant_high_choice" ---
    # create an object to store info about Routine unpleasant_high_choice
    unpleasant_high_choice = data.Routine(
        name='unpleasant_high_choice',
        components=[stressed_high, bored_high, text_unpleasant_high],
    )
    unpleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset stressed_high to account for continued clicks & clear times on/off
    stressed_high.reset()
    # reset bored_high to account for continued clicks & clear times on/off
    bored_high.reset()
    # Run 'Begin Routine' code from code_unpleasant_high
    # Only show this routine if "unpleasant" was selected
    if valence_answer != "unpleasant":
        continueRoutine = False
    # store start times for unpleasant_high_choice
    unpleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    unpleasant_high_choice.tStart = globalClock.getTime(format='float')
    unpleasant_high_choice.status = STARTED
    thisExp.addData('unpleasant_high_choice.started', unpleasant_high_choice.tStart)
    unpleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    unpleasant_high_choiceComponents = unpleasant_high_choice.components
    for thisComponent in unpleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "unpleasant_high_choice" ---
    thisExp.currentRoutine = unpleasant_high_choice
    unpleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *stressed_high* updates
        
        # if stressed_high is starting this frame...
        if stressed_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            stressed_high.frameNStart = frameN  # exact frame index
            stressed_high.tStart = t  # local t and not account for scr refresh
            stressed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stressed_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            stressed_high.status = STARTED
            win.callOnFlip(stressed_high.buttonClock.reset)
            stressed_high.setAutoDraw(True)
        
        # if stressed_high is active this frame...
        if stressed_high.status == STARTED:
            # update params
            pass
            # check whether stressed_high has been pressed
            if stressed_high.isClicked:
                if not stressed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    stressed_high.timesOn.append(stressed_high.buttonClock.getTime())
                    stressed_high.timesOff.append(stressed_high.buttonClock.getTime())
                elif len(stressed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    stressed_high.timesOff[-1] = stressed_high.buttonClock.getTime()
                if not stressed_high.wasClicked:
                    # end routine when stressed_high is clicked
                    continueRoutine = False
                if not stressed_high.wasClicked:
                    # run callback code when stressed_high is clicked
                    pass
        # take note of whether stressed_high was clicked, so that next frame we know if clicks are new
        stressed_high.wasClicked = stressed_high.isClicked and stressed_high.status == STARTED
        # *bored_high* updates
        
        # if bored_high is starting this frame...
        if bored_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            bored_high.frameNStart = frameN  # exact frame index
            bored_high.tStart = t  # local t and not account for scr refresh
            bored_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bored_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bored_high.started')
            # update status
            bored_high.status = STARTED
            win.callOnFlip(bored_high.buttonClock.reset)
            bored_high.setAutoDraw(True)
        
        # if bored_high is active this frame...
        if bored_high.status == STARTED:
            # update params
            pass
            # check whether bored_high has been pressed
            if bored_high.isClicked:
                if not bored_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    bored_high.timesOn.append(bored_high.buttonClock.getTime())
                    bored_high.timesOff.append(bored_high.buttonClock.getTime())
                elif len(bored_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    bored_high.timesOff[-1] = bored_high.buttonClock.getTime()
                if not bored_high.wasClicked:
                    # end routine when bored_high is clicked
                    continueRoutine = False
                if not bored_high.wasClicked:
                    # run callback code when bored_high is clicked
                    pass
        # take note of whether bored_high was clicked, so that next frame we know if clicks are new
        bored_high.wasClicked = bored_high.isClicked and bored_high.status == STARTED
        
        # *text_unpleasant_high* updates
        
        # if text_unpleasant_high is starting this frame...
        if text_unpleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_unpleasant_high.frameNStart = frameN  # exact frame index
            text_unpleasant_high.tStart = t  # local t and not account for scr refresh
            text_unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_unpleasant_high.started')
            # update status
            text_unpleasant_high.status = STARTED
            text_unpleasant_high.setAutoDraw(True)
        
        # if text_unpleasant_high is active this frame...
        if text_unpleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=unpleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            unpleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if unpleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in unpleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "unpleasant_high_choice" ---
    for thisComponent in unpleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for unpleasant_high_choice
    unpleasant_high_choice.tStop = globalClock.getTime(format='float')
    unpleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('unpleasant_high_choice.stopped', unpleasant_high_choice.tStop)
    thisExp.addData('stressed_high.numClicks', stressed_high.numClicks)
    if stressed_high.numClicks:
       thisExp.addData('stressed_high.timesOn', stressed_high.timesOn)
       thisExp.addData('stressed_high.timesOff', stressed_high.timesOff)
    else:
       thisExp.addData('stressed_high.timesOn', "")
       thisExp.addData('stressed_high.timesOff', "")
    thisExp.addData('bored_high.numClicks', bored_high.numClicks)
    if bored_high.numClicks:
       thisExp.addData('bored_high.timesOn', bored_high.timesOn)
       thisExp.addData('bored_high.timesOff', bored_high.timesOff)
    else:
       thisExp.addData('bored_high.timesOn', "")
       thisExp.addData('bored_high.timesOff', "")
    # Run 'End Routine' code from code_unpleasant_high
    if stressed_high.isClicked:
        unpleasant_answer = "stressed"
    elif bored_high.isClicked:
        unpleasant_answer = "bored"
    else:
        unpleasant_answer = "no_response"
    
    thisExp.addData('answ_unpleasant', unpleasant_answer)
    thisExp.nextEntry()
    # the Routine "unpleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "high_reliability_start" ---
    # create an object to store info about Routine high_reliability_start
    high_reliability_start = data.Routine(
        name='high_reliability_start',
        components=[text_high_reliability_start, key_resp_high_reliability_start, text_space_high_reliability_start],
    )
    high_reliability_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_high_reliability_start
    key_resp_high_reliability_start.keys = []
    key_resp_high_reliability_start.rt = []
    _key_resp_high_reliability_start_allKeys = []
    # store start times for high_reliability_start
    high_reliability_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    high_reliability_start.tStart = globalClock.getTime(format='float')
    high_reliability_start.status = STARTED
    thisExp.addData('high_reliability_start.started', high_reliability_start.tStart)
    high_reliability_start.maxDuration = None
    # keep track of which components have finished
    high_reliability_startComponents = high_reliability_start.components
    for thisComponent in high_reliability_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "high_reliability_start" ---
    thisExp.currentRoutine = high_reliability_start
    high_reliability_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_high_reliability_start* updates
        
        # if text_high_reliability_start is starting this frame...
        if text_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_high_reliability_start.frameNStart = frameN  # exact frame index
            text_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_high_reliability_start.status = STARTED
            text_high_reliability_start.setAutoDraw(True)
        
        # if text_high_reliability_start is active this frame...
        if text_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # *key_resp_high_reliability_start* updates
        
        # if key_resp_high_reliability_start is starting this frame...
        if key_resp_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_high_reliability_start.frameNStart = frameN  # exact frame index
            key_resp_high_reliability_start.tStart = t  # local t and not account for scr refresh
            key_resp_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_high_reliability_start.status = STARTED
            # keyboard checking is just starting
            key_resp_high_reliability_start.clock.reset()  # now t=0
        if key_resp_high_reliability_start.status == STARTED:
            theseKeys = key_resp_high_reliability_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_high_reliability_start_allKeys.extend(theseKeys)
            if len(_key_resp_high_reliability_start_allKeys):
                key_resp_high_reliability_start.keys = _key_resp_high_reliability_start_allKeys[-1].name  # just the last key pressed
                key_resp_high_reliability_start.rt = _key_resp_high_reliability_start_allKeys[-1].rt
                key_resp_high_reliability_start.duration = _key_resp_high_reliability_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_high_reliability_start* updates
        
        # if text_space_high_reliability_start is starting this frame...
        if text_space_high_reliability_start.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_high_reliability_start.frameNStart = frameN  # exact frame index
            text_space_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_space_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_high_reliability_start.status = STARTED
            text_space_high_reliability_start.setAutoDraw(True)
        
        # if text_space_high_reliability_start is active this frame...
        if text_space_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=high_reliability_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            high_reliability_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if high_reliability_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in high_reliability_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "high_reliability_start" ---
    for thisComponent in high_reliability_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for high_reliability_start
    high_reliability_start.tStop = globalClock.getTime(format='float')
    high_reliability_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('high_reliability_start.stopped', high_reliability_start.tStop)
    thisExp.nextEntry()
    # the Routine "high_reliability_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    high_2 = data.TrialHandler2(
        name='high_2',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trial_100_2_ecological.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(high_2)  # add the loop to the experiment
    thisHigh_2 = high_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisHigh_2.rgb)
    if thisHigh_2 != None:
        for paramName in thisHigh_2:
            globals()[paramName] = thisHigh_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisHigh_2 in high_2:
        high_2.status = STARTED
        if hasattr(thisHigh_2, 'status'):
            thisHigh_2.status = STARTED
        currentLoop = high_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisHigh_2.rgb)
        if thisHigh_2 != None:
            for paramName in thisHigh_2:
                globals()[paramName] = thisHigh_2[paramName]
        
        # --- Prepare to start Routine "time_high" ---
        # create an object to store info about Routine time_high
        time_high = data.Routine(
            name='time_high',
            components=[],
        )
        time_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for time_high
        time_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_high.tStart = globalClock.getTime(format='float')
        time_high.status = STARTED
        time_high.maxDuration = 1
        # keep track of which components have finished
        time_highComponents = time_high.components
        for thisComponent in time_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "time_high" ---
        thisExp.currentRoutine = time_high
        time_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_2, 'status') and thisHigh_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > time_high.maxDuration-frameTolerance:
                time_high.maxDurationReached = True
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=time_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                time_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if time_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in time_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "time_high" ---
        for thisComponent in time_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for time_high
        time_high.tStop = globalClock.getTime(format='float')
        time_high.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if time_high.maxDurationReached:
            routineTimer.addTime(-time_high.maxDuration)
        elif time_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fixation_cross_look],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_START | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_2, 'status') and thisHigh_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_look* updates
            
            # if fixation_cross_look is starting this frame...
            if fixation_cross_look.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_look.frameNStart = frameN  # exact frame index
                fixation_cross_look.tStart = t  # local t and not account for scr refresh
                fixation_cross_look.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_look, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_look.started')
                # update status
                fixation_cross_look.status = STARTED
                fixation_cross_look.setAutoDraw(True)
            
            # if fixation_cross_look is active this frame...
            if fixation_cross_look.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_look is stopping this frame...
            if fixation_cross_look.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_look.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_look.tStop = t  # not accounting for scr refresh
                    fixation_cross_look.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross_look.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_look.stopped')
                    # update status
                    fixation_cross_look.status = FINISHED
                    fixation_cross_look.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # Run 'End Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_END | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_high" ---
        # create an object to store info about Routine trial_high
        trial_high = data.Routine(
            name='trial_high',
            components=[trial_high_aircraft_1, trial_high_aircraft_2, change_high, maintain_high, support_high],
        )
        trial_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_high_aircraft_1.setText(a1)
        trial_high_aircraft_2.setText(a2)
        # reset change_high to account for continued clicks & clear times on/off
        change_high.reset()
        # reset maintain_high to account for continued clicks & clear times on/off
        maintain_high.reset()
        support_high.setPos((value, -0.3))
        # Run 'Begin Routine' code from code_high
        # Start time
        t_start = globalClock.getTime()
        
        # Interpret automated recommendation
        if value < 0:
            recommendation = "change"
        else:
            recommendation = "maintain"
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Aircraft_1={a1} | "
            f"Aircraft_2={a2} | "
            f"Loss_of_separation={conflict} | "
            f"Automation_recommendation={recommendation} | "
            f"Automation_error={automation_error}"
        )
        outlet.push_sample([marker_info_low])
        
        # Start marker
        marker_start_low = f"START | T{block_trial} | B{block}"
        outlet.push_sample([marker_start_low])
        # store start times for trial_high
        trial_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_high.tStart = globalClock.getTime(format='float')
        trial_high.status = STARTED
        thisExp.addData('trial_high.started', trial_high.tStart)
        trial_high.maxDuration = 3
        # keep track of which components have finished
        trial_highComponents = trial_high.components
        for thisComponent in trial_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_high" ---
        thisExp.currentRoutine = trial_high
        trial_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_2, 'status') and thisHigh_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > trial_high.maxDuration-frameTolerance:
                trial_high.maxDurationReached = True
                continueRoutine = False
            
            # *trial_high_aircraft_1* updates
            
            # if trial_high_aircraft_1 is starting this frame...
            if trial_high_aircraft_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_1.frameNStart = frameN  # exact frame index
                trial_high_aircraft_1.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_high_aircraft_1.started')
                # update status
                trial_high_aircraft_1.status = STARTED
                trial_high_aircraft_1.setAutoDraw(True)
            
            # if trial_high_aircraft_1 is active this frame...
            if trial_high_aircraft_1.status == STARTED:
                # update params
                pass
            
            # *trial_high_aircraft_2* updates
            
            # if trial_high_aircraft_2 is starting this frame...
            if trial_high_aircraft_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_2.frameNStart = frameN  # exact frame index
                trial_high_aircraft_2.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_high_aircraft_2.status = STARTED
                trial_high_aircraft_2.setAutoDraw(True)
            
            # if trial_high_aircraft_2 is active this frame...
            if trial_high_aircraft_2.status == STARTED:
                # update params
                pass
            # *change_high* updates
            
            # if change_high is starting this frame...
            if change_high.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                change_high.frameNStart = frameN  # exact frame index
                change_high.tStart = t  # local t and not account for scr refresh
                change_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(change_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                change_high.status = STARTED
                win.callOnFlip(change_high.buttonClock.reset)
                change_high.setAutoDraw(True)
            
            # if change_high is active this frame...
            if change_high.status == STARTED:
                # update params
                pass
                # check whether change_high has been pressed
                if change_high.isClicked:
                    if not change_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        change_high.timesOn.append(change_high.buttonClock.getTime())
                        change_high.timesOff.append(change_high.buttonClock.getTime())
                    elif len(change_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        change_high.timesOff[-1] = change_high.buttonClock.getTime()
                    if not change_high.wasClicked:
                        # end routine when change_high is clicked
                        continueRoutine = False
                    if not change_high.wasClicked:
                        # run callback code when change_high is clicked
                        pass
            # take note of whether change_high was clicked, so that next frame we know if clicks are new
            change_high.wasClicked = change_high.isClicked and change_high.status == STARTED
            # *maintain_high* updates
            
            # if maintain_high is starting this frame...
            if maintain_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                maintain_high.frameNStart = frameN  # exact frame index
                maintain_high.tStart = t  # local t and not account for scr refresh
                maintain_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maintain_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maintain_high.started')
                # update status
                maintain_high.status = STARTED
                win.callOnFlip(maintain_high.buttonClock.reset)
                maintain_high.setAutoDraw(True)
            
            # if maintain_high is active this frame...
            if maintain_high.status == STARTED:
                # update params
                pass
                # check whether maintain_high has been pressed
                if maintain_high.isClicked:
                    if not maintain_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        maintain_high.timesOn.append(maintain_high.buttonClock.getTime())
                        maintain_high.timesOff.append(maintain_high.buttonClock.getTime())
                    elif len(maintain_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        maintain_high.timesOff[-1] = maintain_high.buttonClock.getTime()
                    if not maintain_high.wasClicked:
                        # end routine when maintain_high is clicked
                        continueRoutine = False
                    if not maintain_high.wasClicked:
                        # run callback code when maintain_high is clicked
                        pass
            # take note of whether maintain_high was clicked, so that next frame we know if clicks are new
            maintain_high.wasClicked = maintain_high.isClicked and maintain_high.status == STARTED
            
            # *support_high* updates
            
            # if support_high is starting this frame...
            if support_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                support_high.frameNStart = frameN  # exact frame index
                support_high.tStart = t  # local t and not account for scr refresh
                support_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(support_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'support_high.started')
                # update status
                support_high.status = STARTED
                support_high.setAutoDraw(True)
            
            # if support_high is active this frame...
            if support_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_high" ---
        for thisComponent in trial_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_high
        trial_high.tStop = globalClock.getTime(format='float')
        trial_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_high.stopped', trial_high.tStop)
        high_2.addData('change_high.numClicks', change_high.numClicks)
        if change_high.numClicks:
           high_2.addData('change_high.timesOn', change_high.timesOn)
           high_2.addData('change_high.timesOff', change_high.timesOff)
        else:
           high_2.addData('change_high.timesOn', "")
           high_2.addData('change_high.timesOff', "")
        high_2.addData('maintain_high.numClicks', maintain_high.numClicks)
        if maintain_high.numClicks:
           high_2.addData('maintain_high.timesOn', maintain_high.timesOn)
           high_2.addData('maintain_high.timesOff', maintain_high.timesOff)
        else:
           high_2.addData('maintain_high.timesOn', "")
           high_2.addData('maintain_high.timesOff', "")
        # Run 'End Routine' code from code_high
        # Detect which button was pressed
        if change_high.isClicked:
            response = "change"
        elif maintain_high.isClicked:
            response = "maintain"
        else:
            response = "no_response"
        
        # Normalize Excel value
        con = str(conflict).strip().lower()
        
        # Evaluate correctness
        if con == "yes" and response == "change":
            result = "Correct action"
            correct = 1
        elif con == "no" and response == "maintain":
            result = "Correct action"
            correct = 1
        elif con == "yes" and response == "maintain":
            result = "Incorrect action"
            correct = 0
        elif con == "no" and response == "change":
            result = "Incorrect action"
            correct = 0
        else:
            result = "Time out"
            correct = 0
        
        # Save data
        thisExp.addData('answer', response)
        thisExp.addData('correct', correct)
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Response={response} | "
            f"Correct={correct}"
        )
        outlet.push_sample([marker_info_low])
        
        # Trial end marker
        marker_end_low = f"END | T{block_trial} | B{block}"
        outlet.push_sample([marker_end_low])
        # the Routine "trial_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "answer_high" ---
        # create an object to store info about Routine answer_high
        answer_high = data.Routine(
            name='answer_high',
            components=[text_answer_high],
        )
        answer_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_answer_high.setText(result)
        # store start times for answer_high
        answer_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        answer_high.tStart = globalClock.getTime(format='float')
        answer_high.status = STARTED
        thisExp.addData('answer_high.started', answer_high.tStart)
        answer_high.maxDuration = None
        # keep track of which components have finished
        answer_highComponents = answer_high.components
        for thisComponent in answer_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "answer_high" ---
        thisExp.currentRoutine = answer_high
        answer_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_2, 'status') and thisHigh_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_answer_high* updates
            
            # if text_answer_high is starting this frame...
            if text_answer_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer_high.frameNStart = frameN  # exact frame index
                text_answer_high.tStart = t  # local t and not account for scr refresh
                text_answer_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_high.started')
                # update status
                text_answer_high.status = STARTED
                text_answer_high.setAutoDraw(True)
            
            # if text_answer_high is active this frame...
            if text_answer_high.status == STARTED:
                # update params
                pass
            
            # if text_answer_high is stopping this frame...
            if text_answer_high.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_answer_high.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_answer_high.tStop = t  # not accounting for scr refresh
                    text_answer_high.tStopRefresh = tThisFlipGlobal  # on global time
                    text_answer_high.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_answer_high.stopped')
                    # update status
                    text_answer_high.status = FINISHED
                    text_answer_high.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=answer_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                answer_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if answer_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in answer_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "answer_high" ---
        for thisComponent in answer_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for answer_high
        answer_high.tStop = globalClock.getTime(format='float')
        answer_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('answer_high.stopped', answer_high.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if answer_high.maxDurationReached:
            routineTimer.addTime(-answer_high.maxDuration)
        elif answer_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_high" ---
        # create an object to store info about Routine question_high
        question_high = data.Routine(
            name='question_high',
            components=[text_high_question, slider_high_question, mouse_high_question],
        )
        question_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_high_question.setText('How much do you trust the automation?')
        slider_high_question.reset()
        # setup some python lists for storing info about the mouse_high_question
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_high_question
        # Reset mouse clicks
        mouse_high_question.clickReset()
        
        # Previous button state
        prevButtonState = mouse_high_question.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        # store start times for question_high
        question_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        question_high.tStart = globalClock.getTime(format='float')
        question_high.status = STARTED
        thisExp.addData('question_high.started', question_high.tStart)
        question_high.maxDuration = None
        # keep track of which components have finished
        question_highComponents = question_high.components
        for thisComponent in question_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_high" ---
        thisExp.currentRoutine = question_high
        question_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_2, 'status') and thisHigh_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_high_question* updates
            
            # if text_high_question is starting this frame...
            if text_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_high_question.frameNStart = frameN  # exact frame index
                text_high_question.tStart = t  # local t and not account for scr refresh
                text_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_high_question.status = STARTED
                text_high_question.setAutoDraw(True)
            
            # if text_high_question is active this frame...
            if text_high_question.status == STARTED:
                # update params
                pass
            
            # *slider_high_question* updates
            
            # if slider_high_question is starting this frame...
            if slider_high_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_high_question.frameNStart = frameN  # exact frame index
                slider_high_question.tStart = t  # local t and not account for scr refresh
                slider_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_high_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_high_question.started')
                # update status
                slider_high_question.status = STARTED
                slider_high_question.setAutoDraw(True)
            
            # if slider_high_question is active this frame...
            if slider_high_question.status == STARTED:
                # update params
                pass
            
            # Check slider_high_question for response to end Routine
            if slider_high_question.getRating() is not None and slider_high_question.status == STARTED:
                continueRoutine = False
            # *mouse_high_question* updates
            
            # if mouse_high_question is starting this frame...
            if mouse_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_high_question.frameNStart = frameN  # exact frame index
                mouse_high_question.tStart = t  # local t and not account for scr refresh
                mouse_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_high_question.status = STARTED
                mouse_high_question.mouseClock.reset()
                prevButtonState = mouse_high_question.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_high_question
            mouse_high_question_pos = mouse_high_question.getPos()
            
            x_min = slider_high_question.pos[0] - slider_high_question.size[0] / 2
            x_max = slider_high_question.pos[0] + slider_high_question.size[0] / 2
            y_min = slider_high_question.pos[1] - 0.1
            y_max = slider_high_question.pos[1] + 0.1
            
            # Move marker when hovering over the slider
            if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                slider_high_question.markerPos = slider_high_question._posToRating(mouse_high_question_pos)
            
            buttonState = mouse_high_question.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                    slider_high_question.rating = slider_high_question.markerPos
                    continueRoutine = False
            
            prevButtonState = buttonState
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=question_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                question_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if question_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in question_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_high" ---
        for thisComponent in question_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for question_high
        question_high.tStop = globalClock.getTime(format='float')
        question_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('question_high.stopped', question_high.tStop)
        # store data for high_2 (TrialHandler)
        # Run 'End Routine' code from code_high_question
        thisExp.addData('trial', block_trial)
        thisExp.addData('answ_high_question', slider_high_question.markerPos)
        
        if slider_high_question.markerPos is not None:
            trust_answer = slider_high_question.markerPos
        else:
            trust_answer = "NA"
        
        marker = f"CONF_H | B{block} | T{block_trial} | V={trust_answer}"
        outlet.push_sample([marker])
        # the Routine "question_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisHigh_2 as finished
        if hasattr(thisHigh_2, 'status'):
            thisHigh_2.status = FINISHED
        # if awaiting a pause, pause now
        if high_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            high_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'high_2'
    high_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "SATI_short_high_instructions" ---
    # create an object to store info about Routine SATI_short_high_instructions
    SATI_short_high_instructions = data.Routine(
        name='SATI_short_high_instructions',
        components=[text_SATI_short_high_instructions, key_resp_SATI_short_high_instructions, text_space_SATI_short_high_instructions],
    )
    SATI_short_high_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_SATI_short_high_instructions
    key_resp_SATI_short_high_instructions.keys = []
    key_resp_SATI_short_high_instructions.rt = []
    _key_resp_SATI_short_high_instructions_allKeys = []
    # store start times for SATI_short_high_instructions
    SATI_short_high_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SATI_short_high_instructions.tStart = globalClock.getTime(format='float')
    SATI_short_high_instructions.status = STARTED
    thisExp.addData('SATI_short_high_instructions.started', SATI_short_high_instructions.tStart)
    SATI_short_high_instructions.maxDuration = None
    # keep track of which components have finished
    SATI_short_high_instructionsComponents = SATI_short_high_instructions.components
    for thisComponent in SATI_short_high_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SATI_short_high_instructions" ---
    thisExp.currentRoutine = SATI_short_high_instructions
    SATI_short_high_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_SATI_short_high_instructions* updates
        
        # if text_SATI_short_high_instructions is starting this frame...
        if text_SATI_short_high_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_SATI_short_high_instructions.started')
            # update status
            text_SATI_short_high_instructions.status = STARTED
            text_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_SATI_short_high_instructions is active this frame...
        if text_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_SATI_short_high_instructions* updates
        
        # if key_resp_SATI_short_high_instructions is starting this frame...
        if key_resp_SATI_short_high_instructions.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            key_resp_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_SATI_short_high_instructions.status = STARTED
            # keyboard checking is just starting
            key_resp_SATI_short_high_instructions.clock.reset()  # now t=0
        if key_resp_SATI_short_high_instructions.status == STARTED:
            theseKeys = key_resp_SATI_short_high_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_SATI_short_high_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_SATI_short_high_instructions_allKeys):
                key_resp_SATI_short_high_instructions.keys = _key_resp_SATI_short_high_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_SATI_short_high_instructions.rt = _key_resp_SATI_short_high_instructions_allKeys[-1].rt
                key_resp_SATI_short_high_instructions.duration = _key_resp_SATI_short_high_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_SATI_short_high_instructions* updates
        
        # if text_space_SATI_short_high_instructions is starting this frame...
        if text_space_SATI_short_high_instructions.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_space_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_space_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_SATI_short_high_instructions.status = STARTED
            text_space_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_space_SATI_short_high_instructions is active this frame...
        if text_space_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SATI_short_high_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            SATI_short_high_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if SATI_short_high_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in SATI_short_high_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SATI_short_high_instructions" ---
    for thisComponent in SATI_short_high_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SATI_short_high_instructions
    SATI_short_high_instructions.tStop = globalClock.getTime(format='float')
    SATI_short_high_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SATI_short_high_instructions.stopped', SATI_short_high_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "SATI_short_high_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    SATI_high_2 = data.TrialHandler2(
        name='SATI_high_2',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('SATI.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(SATI_high_2)  # add the loop to the experiment
    thisSATI_high_2 = SATI_high_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSATI_high_2.rgb)
    if thisSATI_high_2 != None:
        for paramName in thisSATI_high_2:
            globals()[paramName] = thisSATI_high_2[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSATI_high_2 in SATI_high_2:
        SATI_high_2.status = STARTED
        if hasattr(thisSATI_high_2, 'status'):
            thisSATI_high_2.status = STARTED
        currentLoop = SATI_high_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSATI_high_2.rgb)
        if thisSATI_high_2 != None:
            for paramName in thisSATI_high_2:
                globals()[paramName] = thisSATI_high_2[paramName]
        
        # --- Prepare to start Routine "SATI_short_high" ---
        # create an object to store info about Routine SATI_short_high
        SATI_short_high = data.Routine(
            name='SATI_short_high',
            components=[text_SATI_short_high, text_SATI_short_high_clarification, slider_SATI_short_high, mouse_SATI_short_high, prog_short_high],
        )
        SATI_short_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_SATI_short_high.setText(Question)
        text_SATI_short_high_clarification.setText(Clarification)
        slider_SATI_short_high.reset()
        # setup some python lists for storing info about the mouse_SATI_short_high
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_SATI_short_high
        # Reset mouse clicks
        mouse_SATI_short_high.clickReset()
        
        # Previous button state (important for detecting a clean click)
        prevButtonState = mouse_SATI_short_high.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_short_high.setProgress((block_trial - 1) / 6)
        # store start times for SATI_short_high
        SATI_short_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SATI_short_high.tStart = globalClock.getTime(format='float')
        SATI_short_high.status = STARTED
        thisExp.addData('SATI_short_high.started', SATI_short_high.tStart)
        SATI_short_high.maxDuration = None
        # keep track of which components have finished
        SATI_short_highComponents = SATI_short_high.components
        for thisComponent in SATI_short_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SATI_short_high" ---
        thisExp.currentRoutine = SATI_short_high
        SATI_short_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSATI_high_2, 'status') and thisSATI_high_2.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_SATI_short_high* updates
            
            # if text_SATI_short_high is starting this frame...
            if text_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high.frameNStart = frameN  # exact frame index
                text_SATI_short_high.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high.status = STARTED
                text_SATI_short_high.setAutoDraw(True)
            
            # if text_SATI_short_high is active this frame...
            if text_SATI_short_high.status == STARTED:
                # update params
                pass
            
            # *text_SATI_short_high_clarification* updates
            
            # if text_SATI_short_high_clarification is starting this frame...
            if text_SATI_short_high_clarification.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high_clarification.frameNStart = frameN  # exact frame index
                text_SATI_short_high_clarification.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high_clarification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high_clarification, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high_clarification.status = STARTED
                text_SATI_short_high_clarification.setAutoDraw(True)
            
            # if text_SATI_short_high_clarification is active this frame...
            if text_SATI_short_high_clarification.status == STARTED:
                # update params
                pass
            
            # *slider_SATI_short_high* updates
            
            # if slider_SATI_short_high is starting this frame...
            if slider_SATI_short_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_SATI_short_high.frameNStart = frameN  # exact frame index
                slider_SATI_short_high.tStart = t  # local t and not account for scr refresh
                slider_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SATI_short_high.started')
                # update status
                slider_SATI_short_high.status = STARTED
                slider_SATI_short_high.setAutoDraw(True)
            
            # if slider_SATI_short_high is active this frame...
            if slider_SATI_short_high.status == STARTED:
                # update params
                pass
            # *mouse_SATI_short_high* updates
            
            # if mouse_SATI_short_high is starting this frame...
            if mouse_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_SATI_short_high.frameNStart = frameN  # exact frame index
                mouse_SATI_short_high.tStart = t  # local t and not account for scr refresh
                mouse_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_SATI_short_high.status = STARTED
                mouse_SATI_short_high.mouseClock.reset()
                prevButtonState = mouse_SATI_short_high.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_SATI_short_high
            mouse_pos = mouse_SATI_short_high.getPos()
            
            x_min = slider_SATI_short_high.pos[0] - slider_SATI_short_high.size[0] / 2
            x_max = slider_SATI_short_high.pos[0] + slider_SATI_short_high.size[0] / 2
            y_min = slider_SATI_short_high.pos[1] - 0.1
            y_max = slider_SATI_short_high.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_SATI_short_high.markerPos = slider_SATI_short_high._posToRating(mouse_pos)
            
            buttonState = mouse_SATI_short_high.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_SATI_short_high.rating = slider_SATI_short_high.markerPos
                    responseSelected = True
            
            # Wait until the mouse has been released for 0.8 s
            if responseSelected:
                if buttonState == 0:
                    if releaseStartTime is None:
                        releaseStartTime = t
                    elif t - releaseStartTime >= releaseWaitTime:
                        continueRoutine = False
                else:
                    releaseStartTime = None
            
            prevButtonState = buttonState
            
            # *prog_short_high* updates
            
            # if prog_short_high is starting this frame...
            if prog_short_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_short_high.frameNStart = frameN  # exact frame index
                prog_short_high.tStart = t  # local t and not account for scr refresh
                prog_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_short_high.started')
                # update status
                prog_short_high.status = STARTED
                prog_short_high.setAutoDraw(True)
            
            # if prog_short_high is active this frame...
            if prog_short_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=SATI_short_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                SATI_short_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if SATI_short_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in SATI_short_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SATI_short_high" ---
        for thisComponent in SATI_short_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SATI_short_high
        SATI_short_high.tStop = globalClock.getTime(format='float')
        SATI_short_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SATI_short_high.stopped', SATI_short_high.tStop)
        # store data for SATI_high_2 (TrialHandler)
        # Run 'End Routine' code from code_SATI_short_high
        if slider_SATI_short_high.markerPos is not None:
            thisExp.addData('answ_SATI_short_high', slider_SATI_short_high.markerPos)
        else:
            thisExp.addData('answ_SATI_short_high', 'NA')
        # the Routine "SATI_short_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisSATI_high_2 as finished
        if hasattr(thisSATI_high_2, 'status'):
            thisSATI_high_2.status = FINISHED
        # if awaiting a pause, pause now
        if SATI_high_2.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            SATI_high_2.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'SATI_high_2'
    SATI_high_2.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "how_you_feel_high" ---
    # create an object to store info about Routine how_you_feel_high
    how_you_feel_high = data.Routine(
        name='how_you_feel_high',
        components=[pleasant_high, unpleasant_high, text_how_you_feel_high],
    )
    how_you_feel_high.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset pleasant_high to account for continued clicks & clear times on/off
    pleasant_high.reset()
    # reset unpleasant_high to account for continued clicks & clear times on/off
    unpleasant_high.reset()
    # store start times for how_you_feel_high
    how_you_feel_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    how_you_feel_high.tStart = globalClock.getTime(format='float')
    how_you_feel_high.status = STARTED
    thisExp.addData('how_you_feel_high.started', how_you_feel_high.tStart)
    how_you_feel_high.maxDuration = None
    # keep track of which components have finished
    how_you_feel_highComponents = how_you_feel_high.components
    for thisComponent in how_you_feel_high.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "how_you_feel_high" ---
    thisExp.currentRoutine = how_you_feel_high
    how_you_feel_high.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pleasant_high* updates
        
        # if pleasant_high is starting this frame...
        if pleasant_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            pleasant_high.frameNStart = frameN  # exact frame index
            pleasant_high.tStart = t  # local t and not account for scr refresh
            pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            pleasant_high.status = STARTED
            win.callOnFlip(pleasant_high.buttonClock.reset)
            pleasant_high.setAutoDraw(True)
        
        # if pleasant_high is active this frame...
        if pleasant_high.status == STARTED:
            # update params
            pass
            # check whether pleasant_high has been pressed
            if pleasant_high.isClicked:
                if not pleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    pleasant_high.timesOn.append(pleasant_high.buttonClock.getTime())
                    pleasant_high.timesOff.append(pleasant_high.buttonClock.getTime())
                elif len(pleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    pleasant_high.timesOff[-1] = pleasant_high.buttonClock.getTime()
                if not pleasant_high.wasClicked:
                    # end routine when pleasant_high is clicked
                    continueRoutine = False
                if not pleasant_high.wasClicked:
                    # run callback code when pleasant_high is clicked
                    pass
        # take note of whether pleasant_high was clicked, so that next frame we know if clicks are new
        pleasant_high.wasClicked = pleasant_high.isClicked and pleasant_high.status == STARTED
        # *unpleasant_high* updates
        
        # if unpleasant_high is starting this frame...
        if unpleasant_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            unpleasant_high.frameNStart = frameN  # exact frame index
            unpleasant_high.tStart = t  # local t and not account for scr refresh
            unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'unpleasant_high.started')
            # update status
            unpleasant_high.status = STARTED
            win.callOnFlip(unpleasant_high.buttonClock.reset)
            unpleasant_high.setAutoDraw(True)
        
        # if unpleasant_high is active this frame...
        if unpleasant_high.status == STARTED:
            # update params
            pass
            # check whether unpleasant_high has been pressed
            if unpleasant_high.isClicked:
                if not unpleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    unpleasant_high.timesOn.append(unpleasant_high.buttonClock.getTime())
                    unpleasant_high.timesOff.append(unpleasant_high.buttonClock.getTime())
                elif len(unpleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    unpleasant_high.timesOff[-1] = unpleasant_high.buttonClock.getTime()
                if not unpleasant_high.wasClicked:
                    # end routine when unpleasant_high is clicked
                    continueRoutine = False
                if not unpleasant_high.wasClicked:
                    # run callback code when unpleasant_high is clicked
                    pass
        # take note of whether unpleasant_high was clicked, so that next frame we know if clicks are new
        unpleasant_high.wasClicked = unpleasant_high.isClicked and unpleasant_high.status == STARTED
        
        # *text_how_you_feel_high* updates
        
        # if text_how_you_feel_high is starting this frame...
        if text_how_you_feel_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_how_you_feel_high.frameNStart = frameN  # exact frame index
            text_how_you_feel_high.tStart = t  # local t and not account for scr refresh
            text_how_you_feel_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_how_you_feel_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_how_you_feel_high.started')
            # update status
            text_how_you_feel_high.status = STARTED
            text_how_you_feel_high.setAutoDraw(True)
        
        # if text_how_you_feel_high is active this frame...
        if text_how_you_feel_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=how_you_feel_high,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            how_you_feel_high.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if how_you_feel_high.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in how_you_feel_high.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "how_you_feel_high" ---
    for thisComponent in how_you_feel_high.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for how_you_feel_high
    how_you_feel_high.tStop = globalClock.getTime(format='float')
    how_you_feel_high.tStopRefresh = tThisFlipGlobal
    thisExp.addData('how_you_feel_high.stopped', how_you_feel_high.tStop)
    thisExp.addData('pleasant_high.numClicks', pleasant_high.numClicks)
    if pleasant_high.numClicks:
       thisExp.addData('pleasant_high.timesOn', pleasant_high.timesOn)
       thisExp.addData('pleasant_high.timesOff', pleasant_high.timesOff)
    else:
       thisExp.addData('pleasant_high.timesOn', "")
       thisExp.addData('pleasant_high.timesOff', "")
    thisExp.addData('unpleasant_high.numClicks', unpleasant_high.numClicks)
    if unpleasant_high.numClicks:
       thisExp.addData('unpleasant_high.timesOn', unpleasant_high.timesOn)
       thisExp.addData('unpleasant_high.timesOff', unpleasant_high.timesOff)
    else:
       thisExp.addData('unpleasant_high.timesOn', "")
       thisExp.addData('unpleasant_high.timesOff', "")
    # Run 'End Routine' code from code_how_you_feel_high
    if pleasant_high.isClicked:
        valence_answer = "pleasant"
    elif unpleasant_high.isClicked:
        valence_answer = "unpleasant"
    else:
        valence_answer = "no_response"
    
    thisExp.addData('answ_valence', valence_answer)
    thisExp.nextEntry()
    # the Routine "how_you_feel_high" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_high_choice" ---
    # create an object to store info about Routine pleasant_high_choice
    pleasant_high_choice = data.Routine(
        name='pleasant_high_choice',
        components=[excited_high, relaxed_high, text_pleasant_high],
    )
    pleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset excited_high to account for continued clicks & clear times on/off
    excited_high.reset()
    # reset relaxed_high to account for continued clicks & clear times on/off
    relaxed_high.reset()
    # Run 'Begin Routine' code from code_pleasant_high
    # Only show this routine if "pleasant" was selected
    if valence_answer != "pleasant":
        continueRoutine = False
    # store start times for pleasant_high_choice
    pleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pleasant_high_choice.tStart = globalClock.getTime(format='float')
    pleasant_high_choice.status = STARTED
    thisExp.addData('pleasant_high_choice.started', pleasant_high_choice.tStart)
    pleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    pleasant_high_choiceComponents = pleasant_high_choice.components
    for thisComponent in pleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_high_choice" ---
    thisExp.currentRoutine = pleasant_high_choice
    pleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *excited_high* updates
        
        # if excited_high is starting this frame...
        if excited_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            excited_high.frameNStart = frameN  # exact frame index
            excited_high.tStart = t  # local t and not account for scr refresh
            excited_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(excited_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            excited_high.status = STARTED
            win.callOnFlip(excited_high.buttonClock.reset)
            excited_high.setAutoDraw(True)
        
        # if excited_high is active this frame...
        if excited_high.status == STARTED:
            # update params
            pass
            # check whether excited_high has been pressed
            if excited_high.isClicked:
                if not excited_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    excited_high.timesOn.append(excited_high.buttonClock.getTime())
                    excited_high.timesOff.append(excited_high.buttonClock.getTime())
                elif len(excited_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    excited_high.timesOff[-1] = excited_high.buttonClock.getTime()
                if not excited_high.wasClicked:
                    # end routine when excited_high is clicked
                    continueRoutine = False
                if not excited_high.wasClicked:
                    # run callback code when excited_high is clicked
                    pass
        # take note of whether excited_high was clicked, so that next frame we know if clicks are new
        excited_high.wasClicked = excited_high.isClicked and excited_high.status == STARTED
        # *relaxed_high* updates
        
        # if relaxed_high is starting this frame...
        if relaxed_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            relaxed_high.frameNStart = frameN  # exact frame index
            relaxed_high.tStart = t  # local t and not account for scr refresh
            relaxed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(relaxed_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'relaxed_high.started')
            # update status
            relaxed_high.status = STARTED
            win.callOnFlip(relaxed_high.buttonClock.reset)
            relaxed_high.setAutoDraw(True)
        
        # if relaxed_high is active this frame...
        if relaxed_high.status == STARTED:
            # update params
            pass
            # check whether relaxed_high has been pressed
            if relaxed_high.isClicked:
                if not relaxed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    relaxed_high.timesOn.append(relaxed_high.buttonClock.getTime())
                    relaxed_high.timesOff.append(relaxed_high.buttonClock.getTime())
                elif len(relaxed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    relaxed_high.timesOff[-1] = relaxed_high.buttonClock.getTime()
                if not relaxed_high.wasClicked:
                    # end routine when relaxed_high is clicked
                    continueRoutine = False
                if not relaxed_high.wasClicked:
                    # run callback code when relaxed_high is clicked
                    pass
        # take note of whether relaxed_high was clicked, so that next frame we know if clicks are new
        relaxed_high.wasClicked = relaxed_high.isClicked and relaxed_high.status == STARTED
        
        # *text_pleasant_high* updates
        
        # if text_pleasant_high is starting this frame...
        if text_pleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pleasant_high.frameNStart = frameN  # exact frame index
            text_pleasant_high.tStart = t  # local t and not account for scr refresh
            text_pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pleasant_high.started')
            # update status
            text_pleasant_high.status = STARTED
            text_pleasant_high.setAutoDraw(True)
        
        # if text_pleasant_high is active this frame...
        if text_pleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=pleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            pleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if pleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in pleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_high_choice" ---
    for thisComponent in pleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pleasant_high_choice
    pleasant_high_choice.tStop = globalClock.getTime(format='float')
    pleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pleasant_high_choice.stopped', pleasant_high_choice.tStop)
    thisExp.addData('excited_high.numClicks', excited_high.numClicks)
    if excited_high.numClicks:
       thisExp.addData('excited_high.timesOn', excited_high.timesOn)
       thisExp.addData('excited_high.timesOff', excited_high.timesOff)
    else:
       thisExp.addData('excited_high.timesOn', "")
       thisExp.addData('excited_high.timesOff', "")
    thisExp.addData('relaxed_high.numClicks', relaxed_high.numClicks)
    if relaxed_high.numClicks:
       thisExp.addData('relaxed_high.timesOn', relaxed_high.timesOn)
       thisExp.addData('relaxed_high.timesOff', relaxed_high.timesOff)
    else:
       thisExp.addData('relaxed_high.timesOn', "")
       thisExp.addData('relaxed_high.timesOff', "")
    # Run 'End Routine' code from code_pleasant_high
    if excited_high.isClicked:
        pleasant_answer = "excited"
    elif relaxed_high.isClicked:
        pleasant_answer = "relaxed"
    else:
        pleasant_answer = "no_response"
    
    thisExp.addData('answ_pleasant', pleasant_answer)
    thisExp.nextEntry()
    # the Routine "pleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "unpleasant_high_choice" ---
    # create an object to store info about Routine unpleasant_high_choice
    unpleasant_high_choice = data.Routine(
        name='unpleasant_high_choice',
        components=[stressed_high, bored_high, text_unpleasant_high],
    )
    unpleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset stressed_high to account for continued clicks & clear times on/off
    stressed_high.reset()
    # reset bored_high to account for continued clicks & clear times on/off
    bored_high.reset()
    # Run 'Begin Routine' code from code_unpleasant_high
    # Only show this routine if "unpleasant" was selected
    if valence_answer != "unpleasant":
        continueRoutine = False
    # store start times for unpleasant_high_choice
    unpleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    unpleasant_high_choice.tStart = globalClock.getTime(format='float')
    unpleasant_high_choice.status = STARTED
    thisExp.addData('unpleasant_high_choice.started', unpleasant_high_choice.tStart)
    unpleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    unpleasant_high_choiceComponents = unpleasant_high_choice.components
    for thisComponent in unpleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "unpleasant_high_choice" ---
    thisExp.currentRoutine = unpleasant_high_choice
    unpleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *stressed_high* updates
        
        # if stressed_high is starting this frame...
        if stressed_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            stressed_high.frameNStart = frameN  # exact frame index
            stressed_high.tStart = t  # local t and not account for scr refresh
            stressed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stressed_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            stressed_high.status = STARTED
            win.callOnFlip(stressed_high.buttonClock.reset)
            stressed_high.setAutoDraw(True)
        
        # if stressed_high is active this frame...
        if stressed_high.status == STARTED:
            # update params
            pass
            # check whether stressed_high has been pressed
            if stressed_high.isClicked:
                if not stressed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    stressed_high.timesOn.append(stressed_high.buttonClock.getTime())
                    stressed_high.timesOff.append(stressed_high.buttonClock.getTime())
                elif len(stressed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    stressed_high.timesOff[-1] = stressed_high.buttonClock.getTime()
                if not stressed_high.wasClicked:
                    # end routine when stressed_high is clicked
                    continueRoutine = False
                if not stressed_high.wasClicked:
                    # run callback code when stressed_high is clicked
                    pass
        # take note of whether stressed_high was clicked, so that next frame we know if clicks are new
        stressed_high.wasClicked = stressed_high.isClicked and stressed_high.status == STARTED
        # *bored_high* updates
        
        # if bored_high is starting this frame...
        if bored_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            bored_high.frameNStart = frameN  # exact frame index
            bored_high.tStart = t  # local t and not account for scr refresh
            bored_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bored_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bored_high.started')
            # update status
            bored_high.status = STARTED
            win.callOnFlip(bored_high.buttonClock.reset)
            bored_high.setAutoDraw(True)
        
        # if bored_high is active this frame...
        if bored_high.status == STARTED:
            # update params
            pass
            # check whether bored_high has been pressed
            if bored_high.isClicked:
                if not bored_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    bored_high.timesOn.append(bored_high.buttonClock.getTime())
                    bored_high.timesOff.append(bored_high.buttonClock.getTime())
                elif len(bored_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    bored_high.timesOff[-1] = bored_high.buttonClock.getTime()
                if not bored_high.wasClicked:
                    # end routine when bored_high is clicked
                    continueRoutine = False
                if not bored_high.wasClicked:
                    # run callback code when bored_high is clicked
                    pass
        # take note of whether bored_high was clicked, so that next frame we know if clicks are new
        bored_high.wasClicked = bored_high.isClicked and bored_high.status == STARTED
        
        # *text_unpleasant_high* updates
        
        # if text_unpleasant_high is starting this frame...
        if text_unpleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_unpleasant_high.frameNStart = frameN  # exact frame index
            text_unpleasant_high.tStart = t  # local t and not account for scr refresh
            text_unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_unpleasant_high.started')
            # update status
            text_unpleasant_high.status = STARTED
            text_unpleasant_high.setAutoDraw(True)
        
        # if text_unpleasant_high is active this frame...
        if text_unpleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=unpleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            unpleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if unpleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in unpleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "unpleasant_high_choice" ---
    for thisComponent in unpleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for unpleasant_high_choice
    unpleasant_high_choice.tStop = globalClock.getTime(format='float')
    unpleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('unpleasant_high_choice.stopped', unpleasant_high_choice.tStop)
    thisExp.addData('stressed_high.numClicks', stressed_high.numClicks)
    if stressed_high.numClicks:
       thisExp.addData('stressed_high.timesOn', stressed_high.timesOn)
       thisExp.addData('stressed_high.timesOff', stressed_high.timesOff)
    else:
       thisExp.addData('stressed_high.timesOn', "")
       thisExp.addData('stressed_high.timesOff', "")
    thisExp.addData('bored_high.numClicks', bored_high.numClicks)
    if bored_high.numClicks:
       thisExp.addData('bored_high.timesOn', bored_high.timesOn)
       thisExp.addData('bored_high.timesOff', bored_high.timesOff)
    else:
       thisExp.addData('bored_high.timesOn', "")
       thisExp.addData('bored_high.timesOff', "")
    # Run 'End Routine' code from code_unpleasant_high
    if stressed_high.isClicked:
        unpleasant_answer = "stressed"
    elif bored_high.isClicked:
        unpleasant_answer = "bored"
    else:
        unpleasant_answer = "no_response"
    
    thisExp.addData('answ_unpleasant', unpleasant_answer)
    thisExp.nextEntry()
    # the Routine "unpleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "low_reliability_start" ---
    # create an object to store info about Routine low_reliability_start
    low_reliability_start = data.Routine(
        name='low_reliability_start',
        components=[text_low_reliability_start, key_resp_low_reliability_start, text_space_low_reliability_start],
    )
    low_reliability_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_low_reliability_start
    key_resp_low_reliability_start.keys = []
    key_resp_low_reliability_start.rt = []
    _key_resp_low_reliability_start_allKeys = []
    # store start times for low_reliability_start
    low_reliability_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    low_reliability_start.tStart = globalClock.getTime(format='float')
    low_reliability_start.status = STARTED
    thisExp.addData('low_reliability_start.started', low_reliability_start.tStart)
    low_reliability_start.maxDuration = None
    # keep track of which components have finished
    low_reliability_startComponents = low_reliability_start.components
    for thisComponent in low_reliability_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "low_reliability_start" ---
    thisExp.currentRoutine = low_reliability_start
    low_reliability_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_low_reliability_start* updates
        
        # if text_low_reliability_start is starting this frame...
        if text_low_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_low_reliability_start.frameNStart = frameN  # exact frame index
            text_low_reliability_start.tStart = t  # local t and not account for scr refresh
            text_low_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_low_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_low_reliability_start.status = STARTED
            text_low_reliability_start.setAutoDraw(True)
        
        # if text_low_reliability_start is active this frame...
        if text_low_reliability_start.status == STARTED:
            # update params
            pass
        
        # *key_resp_low_reliability_start* updates
        
        # if key_resp_low_reliability_start is starting this frame...
        if key_resp_low_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_low_reliability_start.frameNStart = frameN  # exact frame index
            key_resp_low_reliability_start.tStart = t  # local t and not account for scr refresh
            key_resp_low_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_low_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_low_reliability_start.status = STARTED
            # keyboard checking is just starting
            key_resp_low_reliability_start.clock.reset()  # now t=0
        if key_resp_low_reliability_start.status == STARTED:
            theseKeys = key_resp_low_reliability_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_low_reliability_start_allKeys.extend(theseKeys)
            if len(_key_resp_low_reliability_start_allKeys):
                key_resp_low_reliability_start.keys = _key_resp_low_reliability_start_allKeys[-1].name  # just the last key pressed
                key_resp_low_reliability_start.rt = _key_resp_low_reliability_start_allKeys[-1].rt
                key_resp_low_reliability_start.duration = _key_resp_low_reliability_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_low_reliability_start* updates
        
        # if text_space_low_reliability_start is starting this frame...
        if text_space_low_reliability_start.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_low_reliability_start.frameNStart = frameN  # exact frame index
            text_space_low_reliability_start.tStart = t  # local t and not account for scr refresh
            text_space_low_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_low_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_low_reliability_start.status = STARTED
            text_space_low_reliability_start.setAutoDraw(True)
        
        # if text_space_low_reliability_start is active this frame...
        if text_space_low_reliability_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=low_reliability_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            low_reliability_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if low_reliability_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in low_reliability_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "low_reliability_start" ---
    for thisComponent in low_reliability_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for low_reliability_start
    low_reliability_start.tStop = globalClock.getTime(format='float')
    low_reliability_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('low_reliability_start.stopped', low_reliability_start.tStop)
    thisExp.nextEntry()
    # the Routine "low_reliability_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    low = data.TrialHandler2(
        name='low',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trial_50_ecological.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(low)  # add the loop to the experiment
    thisLow = low.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLow.rgb)
    if thisLow != None:
        for paramName in thisLow:
            globals()[paramName] = thisLow[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLow in low:
        low.status = STARTED
        if hasattr(thisLow, 'status'):
            thisLow.status = STARTED
        currentLoop = low
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLow.rgb)
        if thisLow != None:
            for paramName in thisLow:
                globals()[paramName] = thisLow[paramName]
        
        # --- Prepare to start Routine "time_low" ---
        # create an object to store info about Routine time_low
        time_low = data.Routine(
            name='time_low',
            components=[],
        )
        time_low.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for time_low
        time_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_low.tStart = globalClock.getTime(format='float')
        time_low.status = STARTED
        thisExp.addData('time_low.started', time_low.tStart)
        time_low.maxDuration = 1
        # keep track of which components have finished
        time_lowComponents = time_low.components
        for thisComponent in time_low.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "time_low" ---
        thisExp.currentRoutine = time_low
        time_low.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisLow, 'status') and thisLow.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > time_low.maxDuration-frameTolerance:
                time_low.maxDurationReached = True
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=time_low,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                time_low.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if time_low.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in time_low.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "time_low" ---
        for thisComponent in time_low.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for time_low
        time_low.tStop = globalClock.getTime(format='float')
        time_low.tStopRefresh = tThisFlipGlobal
        thisExp.addData('time_low.stopped', time_low.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if time_low.maxDurationReached:
            routineTimer.addTime(-time_low.maxDuration)
        elif time_low.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fixation_cross_look],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_START | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisLow, 'status') and thisLow.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_look* updates
            
            # if fixation_cross_look is starting this frame...
            if fixation_cross_look.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_look.frameNStart = frameN  # exact frame index
                fixation_cross_look.tStart = t  # local t and not account for scr refresh
                fixation_cross_look.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_look, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_look.started')
                # update status
                fixation_cross_look.status = STARTED
                fixation_cross_look.setAutoDraw(True)
            
            # if fixation_cross_look is active this frame...
            if fixation_cross_look.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_look is stopping this frame...
            if fixation_cross_look.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_look.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_look.tStop = t  # not accounting for scr refresh
                    fixation_cross_look.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross_look.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_look.stopped')
                    # update status
                    fixation_cross_look.status = FINISHED
                    fixation_cross_look.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # Run 'End Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_END | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_low" ---
        # create an object to store info about Routine trial_low
        trial_low = data.Routine(
            name='trial_low',
            components=[trial_low_aircraft_1, trial_low_aircraft_2, change_low, maintain_low, support_low],
        )
        trial_low.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_low_aircraft_1.setText(a1)
        trial_low_aircraft_2.setText(a2)
        # reset change_low to account for continued clicks & clear times on/off
        change_low.reset()
        # reset maintain_low to account for continued clicks & clear times on/off
        maintain_low.reset()
        support_low.setPos((value, -0.3))
        # Run 'Begin Routine' code from code_low
        # Start time
        t_start = globalClock.getTime()
        
        # Interpret automated recommendation
        if value < 0:
            recommendation = "change"
        else:
            recommendation = "maintain"
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Aircraft_1={a1} | "
            f"Aircraft_2={a2} | "
            f"Loss_of_separation={conflict} | "
            f"Automation_recommendation={recommendation} | "
            f"Automation_error={automation_error}"
        )
        outlet.push_sample([marker_info_low])
        
        # Trial start marker
        marker_start_low = f"START | T{block_trial} | B{block}"
        outlet.push_sample([marker_start_low])
        # store start times for trial_low
        trial_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_low.tStart = globalClock.getTime(format='float')
        trial_low.status = STARTED
        thisExp.addData('trial_low.started', trial_low.tStart)
        trial_low.maxDuration = 3
        # keep track of which components have finished
        trial_lowComponents = trial_low.components
        for thisComponent in trial_low.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_low" ---
        thisExp.currentRoutine = trial_low
        trial_low.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLow, 'status') and thisLow.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > trial_low.maxDuration-frameTolerance:
                trial_low.maxDurationReached = True
                continueRoutine = False
            
            # *trial_low_aircraft_1* updates
            
            # if trial_low_aircraft_1 is starting this frame...
            if trial_low_aircraft_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_low_aircraft_1.frameNStart = frameN  # exact frame index
                trial_low_aircraft_1.tStart = t  # local t and not account for scr refresh
                trial_low_aircraft_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_low_aircraft_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_low_aircraft_1.started')
                # update status
                trial_low_aircraft_1.status = STARTED
                trial_low_aircraft_1.setAutoDraw(True)
            
            # if trial_low_aircraft_1 is active this frame...
            if trial_low_aircraft_1.status == STARTED:
                # update params
                pass
            
            # *trial_low_aircraft_2* updates
            
            # if trial_low_aircraft_2 is starting this frame...
            if trial_low_aircraft_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                trial_low_aircraft_2.frameNStart = frameN  # exact frame index
                trial_low_aircraft_2.tStart = t  # local t and not account for scr refresh
                trial_low_aircraft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_low_aircraft_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_low_aircraft_2.status = STARTED
                trial_low_aircraft_2.setAutoDraw(True)
            
            # if trial_low_aircraft_2 is active this frame...
            if trial_low_aircraft_2.status == STARTED:
                # update params
                pass
            # *change_low* updates
            
            # if change_low is starting this frame...
            if change_low.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                change_low.frameNStart = frameN  # exact frame index
                change_low.tStart = t  # local t and not account for scr refresh
                change_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(change_low, 'tStartRefresh')  # time at next scr refresh
                # update status
                change_low.status = STARTED
                win.callOnFlip(change_low.buttonClock.reset)
                change_low.setAutoDraw(True)
            
            # if change_low is active this frame...
            if change_low.status == STARTED:
                # update params
                pass
                # check whether change_low has been pressed
                if change_low.isClicked:
                    if not change_low.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        change_low.timesOn.append(change_low.buttonClock.getTime())
                        change_low.timesOff.append(change_low.buttonClock.getTime())
                    elif len(change_low.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        change_low.timesOff[-1] = change_low.buttonClock.getTime()
                    if not change_low.wasClicked:
                        # end routine when change_low is clicked
                        continueRoutine = False
                    if not change_low.wasClicked:
                        # run callback code when change_low is clicked
                        pass
            # take note of whether change_low was clicked, so that next frame we know if clicks are new
            change_low.wasClicked = change_low.isClicked and change_low.status == STARTED
            # *maintain_low* updates
            
            # if maintain_low is starting this frame...
            if maintain_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                maintain_low.frameNStart = frameN  # exact frame index
                maintain_low.tStart = t  # local t and not account for scr refresh
                maintain_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maintain_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maintain_low.started')
                # update status
                maintain_low.status = STARTED
                win.callOnFlip(maintain_low.buttonClock.reset)
                maintain_low.setAutoDraw(True)
            
            # if maintain_low is active this frame...
            if maintain_low.status == STARTED:
                # update params
                pass
                # check whether maintain_low has been pressed
                if maintain_low.isClicked:
                    if not maintain_low.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        maintain_low.timesOn.append(maintain_low.buttonClock.getTime())
                        maintain_low.timesOff.append(maintain_low.buttonClock.getTime())
                    elif len(maintain_low.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        maintain_low.timesOff[-1] = maintain_low.buttonClock.getTime()
                    if not maintain_low.wasClicked:
                        # end routine when maintain_low is clicked
                        continueRoutine = False
                    if not maintain_low.wasClicked:
                        # run callback code when maintain_low is clicked
                        pass
            # take note of whether maintain_low was clicked, so that next frame we know if clicks are new
            maintain_low.wasClicked = maintain_low.isClicked and maintain_low.status == STARTED
            
            # *support_low* updates
            
            # if support_low is starting this frame...
            if support_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                support_low.frameNStart = frameN  # exact frame index
                support_low.tStart = t  # local t and not account for scr refresh
                support_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(support_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'support_low.started')
                # update status
                support_low.status = STARTED
                support_low.setAutoDraw(True)
            
            # if support_low is active this frame...
            if support_low.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_low,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_low.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_low.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_low.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_low" ---
        for thisComponent in trial_low.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_low
        trial_low.tStop = globalClock.getTime(format='float')
        trial_low.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_low.stopped', trial_low.tStop)
        low.addData('change_low.numClicks', change_low.numClicks)
        if change_low.numClicks:
           low.addData('change_low.timesOn', change_low.timesOn)
           low.addData('change_low.timesOff', change_low.timesOff)
        else:
           low.addData('change_low.timesOn', "")
           low.addData('change_low.timesOff', "")
        low.addData('maintain_low.numClicks', maintain_low.numClicks)
        if maintain_low.numClicks:
           low.addData('maintain_low.timesOn', maintain_low.timesOn)
           low.addData('maintain_low.timesOff', maintain_low.timesOff)
        else:
           low.addData('maintain_low.timesOn', "")
           low.addData('maintain_low.timesOff', "")
        # Run 'End Routine' code from code_low
        # Detect which button was pressed
        if change_low.isClicked:
            response = "change"
        elif maintain_low.isClicked:
            response = "maintain"
        else:
            response = "no_response"
        
        # Normalize Excel value
        con = str(conflict).strip().lower()
        
        # Evaluate correctness
        if con == "yes" and response == "change":
            result = "Correct action"
            correct = 1
        elif con == "no" and response == "maintain":
            result = "Correct action"
            correct = 1
        elif con == "yes" and response == "maintain":
            result = "Incorrect action"
            correct = 0
        elif con == "no" and response == "change":
            result = "Incorrect action"
            correct = 0
        else:
            result = "Time out"
            correct = 0
        
        # Save data
        thisExp.addData('answer', response)
        thisExp.addData('correct', correct)
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Response={response} | "
            f"Correct={correct}"
        )
        outlet.push_sample([marker_info_low])
        
        # Trial end marker
        marker_end_low = f"END | T{block_trial} | B{block}"
        outlet.push_sample([marker_end_low])
        # the Routine "trial_low" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "answer_low" ---
        # create an object to store info about Routine answer_low
        answer_low = data.Routine(
            name='answer_low',
            components=[text_answer_low],
        )
        answer_low.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_answer_low.setText(result)
        # store start times for answer_low
        answer_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        answer_low.tStart = globalClock.getTime(format='float')
        answer_low.status = STARTED
        thisExp.addData('answer_low.started', answer_low.tStart)
        answer_low.maxDuration = None
        # keep track of which components have finished
        answer_lowComponents = answer_low.components
        for thisComponent in answer_low.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "answer_low" ---
        thisExp.currentRoutine = answer_low
        answer_low.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisLow, 'status') and thisLow.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_answer_low* updates
            
            # if text_answer_low is starting this frame...
            if text_answer_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer_low.frameNStart = frameN  # exact frame index
                text_answer_low.tStart = t  # local t and not account for scr refresh
                text_answer_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_low.started')
                # update status
                text_answer_low.status = STARTED
                text_answer_low.setAutoDraw(True)
            
            # if text_answer_low is active this frame...
            if text_answer_low.status == STARTED:
                # update params
                pass
            
            # if text_answer_low is stopping this frame...
            if text_answer_low.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_answer_low.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_answer_low.tStop = t  # not accounting for scr refresh
                    text_answer_low.tStopRefresh = tThisFlipGlobal  # on global time
                    text_answer_low.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_answer_low.stopped')
                    # update status
                    text_answer_low.status = FINISHED
                    text_answer_low.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=answer_low,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                answer_low.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if answer_low.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in answer_low.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "answer_low" ---
        for thisComponent in answer_low.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for answer_low
        answer_low.tStop = globalClock.getTime(format='float')
        answer_low.tStopRefresh = tThisFlipGlobal
        thisExp.addData('answer_low.stopped', answer_low.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if answer_low.maxDurationReached:
            routineTimer.addTime(-answer_low.maxDuration)
        elif answer_low.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_low" ---
        # create an object to store info about Routine question_low
        question_low = data.Routine(
            name='question_low',
            components=[text_low_question, slider_low_question, mouse_low_question],
        )
        question_low.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_low_question.setText('How much do you trust the automation?')
        slider_low_question.reset()
        # setup some python lists for storing info about the mouse_low_question
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_low_question
        mouse_low_question.clickReset()
        prevButtonState = mouse_low_question.getPressed()[0]
        readyForClick = False
        # store start times for question_low
        question_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        question_low.tStart = globalClock.getTime(format='float')
        question_low.status = STARTED
        thisExp.addData('question_low.started', question_low.tStart)
        question_low.maxDuration = None
        # keep track of which components have finished
        question_lowComponents = question_low.components
        for thisComponent in question_low.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_low" ---
        thisExp.currentRoutine = question_low
        question_low.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisLow, 'status') and thisLow.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_low_question* updates
            
            # if text_low_question is starting this frame...
            if text_low_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_low_question.frameNStart = frameN  # exact frame index
                text_low_question.tStart = t  # local t and not account for scr refresh
                text_low_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_low_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_low_question.status = STARTED
                text_low_question.setAutoDraw(True)
            
            # if text_low_question is active this frame...
            if text_low_question.status == STARTED:
                # update params
                pass
            
            # *slider_low_question* updates
            
            # if slider_low_question is starting this frame...
            if slider_low_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_low_question.frameNStart = frameN  # exact frame index
                slider_low_question.tStart = t  # local t and not account for scr refresh
                slider_low_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_low_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_low_question.started')
                # update status
                slider_low_question.status = STARTED
                slider_low_question.setAutoDraw(True)
            
            # if slider_low_question is active this frame...
            if slider_low_question.status == STARTED:
                # update params
                pass
            
            # Check slider_low_question for response to end Routine
            if slider_low_question.getRating() is not None and slider_low_question.status == STARTED:
                continueRoutine = False
            # *mouse_low_question* updates
            
            # if mouse_low_question is starting this frame...
            if mouse_low_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_low_question.frameNStart = frameN  # exact frame index
                mouse_low_question.tStart = t  # local t and not account for scr refresh
                mouse_low_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_low_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_low_question.status = STARTED
                mouse_low_question.mouseClock.reset()
                prevButtonState = mouse_low_question.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_low_question
            mouse_low_question_pos = mouse_low_question.getPos()
            
            x_min = slider_low_question.pos[0] - slider_low_question.size[0] / 2
            x_max = slider_low_question.pos[0] + slider_low_question.size[0] / 2
            y_min = slider_low_question.pos[1] - 0.1
            y_max = slider_low_question.pos[1] + 0.1
            
            # Move marker when hovering over the slider
            if x_min <= mouse_low_question_pos[0] <= x_max and y_min <= mouse_low_question_pos[1] <= y_max:
                slider_low_question.markerPos = slider_low_question._posToRating(mouse_low_question_pos)
            
            buttonState = mouse_low_question.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_low_question_pos[0] <= x_max and y_min <= mouse_low_question_pos[1] <= y_max:
                    slider_low_question.rating = slider_low_question.markerPos
                    continueRoutine = False
            
            prevButtonState = buttonState
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=question_low,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                question_low.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if question_low.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in question_low.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_low" ---
        for thisComponent in question_low.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for question_low
        question_low.tStop = globalClock.getTime(format='float')
        question_low.tStopRefresh = tThisFlipGlobal
        thisExp.addData('question_low.stopped', question_low.tStop)
        # store data for low (TrialHandler)
        # Run 'End Routine' code from code_low_question
        thisExp.addData('trial', block_trial)
        thisExp.addData('answ_low_question', slider_low_question.markerPos)
        
        # Get slider response
        if slider_low_question.markerPos is not None:
            trust_answer = slider_low_question.markerPos
        else:
            trust_answer = "NA"
        
        # Marker
        marker = f"CONF_L | B{block} | T{block_trial} | V{trust_answer}"
        outlet.push_sample([marker])
        # the Routine "question_low" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisLow as finished
        if hasattr(thisLow, 'status'):
            thisLow.status = FINISHED
        # if awaiting a pause, pause now
        if low.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            low.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'low'
    low.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "SATI_short_low_instructions" ---
    # create an object to store info about Routine SATI_short_low_instructions
    SATI_short_low_instructions = data.Routine(
        name='SATI_short_low_instructions',
        components=[text_SATI_short_low_instructions, key_resp_SATI_short_low_instructions, text_space_SATI_short_low_instructions],
    )
    SATI_short_low_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_SATI_short_low_instructions
    key_resp_SATI_short_low_instructions.keys = []
    key_resp_SATI_short_low_instructions.rt = []
    _key_resp_SATI_short_low_instructions_allKeys = []
    # store start times for SATI_short_low_instructions
    SATI_short_low_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SATI_short_low_instructions.tStart = globalClock.getTime(format='float')
    SATI_short_low_instructions.status = STARTED
    thisExp.addData('SATI_short_low_instructions.started', SATI_short_low_instructions.tStart)
    SATI_short_low_instructions.maxDuration = None
    # keep track of which components have finished
    SATI_short_low_instructionsComponents = SATI_short_low_instructions.components
    for thisComponent in SATI_short_low_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SATI_short_low_instructions" ---
    thisExp.currentRoutine = SATI_short_low_instructions
    SATI_short_low_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_SATI_short_low_instructions* updates
        
        # if text_SATI_short_low_instructions is starting this frame...
        if text_SATI_short_low_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_SATI_short_low_instructions.frameNStart = frameN  # exact frame index
            text_SATI_short_low_instructions.tStart = t  # local t and not account for scr refresh
            text_SATI_short_low_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_SATI_short_low_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_SATI_short_low_instructions.started')
            # update status
            text_SATI_short_low_instructions.status = STARTED
            text_SATI_short_low_instructions.setAutoDraw(True)
        
        # if text_SATI_short_low_instructions is active this frame...
        if text_SATI_short_low_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_SATI_short_low_instructions* updates
        
        # if key_resp_SATI_short_low_instructions is starting this frame...
        if key_resp_SATI_short_low_instructions.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_SATI_short_low_instructions.frameNStart = frameN  # exact frame index
            key_resp_SATI_short_low_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_SATI_short_low_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_SATI_short_low_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_SATI_short_low_instructions.status = STARTED
            # keyboard checking is just starting
            key_resp_SATI_short_low_instructions.clock.reset()  # now t=0
        if key_resp_SATI_short_low_instructions.status == STARTED:
            theseKeys = key_resp_SATI_short_low_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_SATI_short_low_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_SATI_short_low_instructions_allKeys):
                key_resp_SATI_short_low_instructions.keys = _key_resp_SATI_short_low_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_SATI_short_low_instructions.rt = _key_resp_SATI_short_low_instructions_allKeys[-1].rt
                key_resp_SATI_short_low_instructions.duration = _key_resp_SATI_short_low_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_SATI_short_low_instructions* updates
        
        # if text_space_SATI_short_low_instructions is starting this frame...
        if text_space_SATI_short_low_instructions.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_SATI_short_low_instructions.frameNStart = frameN  # exact frame index
            text_space_SATI_short_low_instructions.tStart = t  # local t and not account for scr refresh
            text_space_SATI_short_low_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_SATI_short_low_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_SATI_short_low_instructions.status = STARTED
            text_space_SATI_short_low_instructions.setAutoDraw(True)
        
        # if text_space_SATI_short_low_instructions is active this frame...
        if text_space_SATI_short_low_instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SATI_short_low_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            SATI_short_low_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if SATI_short_low_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in SATI_short_low_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SATI_short_low_instructions" ---
    for thisComponent in SATI_short_low_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SATI_short_low_instructions
    SATI_short_low_instructions.tStop = globalClock.getTime(format='float')
    SATI_short_low_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SATI_short_low_instructions.stopped', SATI_short_low_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "SATI_short_low_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    SATI_low = data.TrialHandler2(
        name='SATI_low',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('SATI.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(SATI_low)  # add the loop to the experiment
    thisSATI_low = SATI_low.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSATI_low.rgb)
    if thisSATI_low != None:
        for paramName in thisSATI_low:
            globals()[paramName] = thisSATI_low[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSATI_low in SATI_low:
        SATI_low.status = STARTED
        if hasattr(thisSATI_low, 'status'):
            thisSATI_low.status = STARTED
        currentLoop = SATI_low
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSATI_low.rgb)
        if thisSATI_low != None:
            for paramName in thisSATI_low:
                globals()[paramName] = thisSATI_low[paramName]
        
        # --- Prepare to start Routine "SATI_short_low" ---
        # create an object to store info about Routine SATI_short_low
        SATI_short_low = data.Routine(
            name='SATI_short_low',
            components=[text_SATI_short_low, text_SATI_short_low_clarification, slider_SATI_short_low, mouse_SATI_short_low, prog_short_low],
        )
        SATI_short_low.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_SATI_short_low.setText(Question)
        text_SATI_short_low_clarification.setText(Clarification)
        slider_SATI_short_low.reset()
        # setup some python lists for storing info about the mouse_SATI_short_low
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_SATI_short_low
        # Reset mouse clicks
        mouse_SATI_short_low.clickReset()
        
        # Previous button state
        prevButtonState = mouse_SATI_short_low.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_short_low.setProgress((block_trial - 1) / 6)
        # store start times for SATI_short_low
        SATI_short_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SATI_short_low.tStart = globalClock.getTime(format='float')
        SATI_short_low.status = STARTED
        thisExp.addData('SATI_short_low.started', SATI_short_low.tStart)
        SATI_short_low.maxDuration = None
        # keep track of which components have finished
        SATI_short_lowComponents = SATI_short_low.components
        for thisComponent in SATI_short_low.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SATI_short_low" ---
        thisExp.currentRoutine = SATI_short_low
        SATI_short_low.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSATI_low, 'status') and thisSATI_low.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_SATI_short_low* updates
            
            # if text_SATI_short_low is starting this frame...
            if text_SATI_short_low.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_low.frameNStart = frameN  # exact frame index
                text_SATI_short_low.tStart = t  # local t and not account for scr refresh
                text_SATI_short_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_low, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_low.status = STARTED
                text_SATI_short_low.setAutoDraw(True)
            
            # if text_SATI_short_low is active this frame...
            if text_SATI_short_low.status == STARTED:
                # update params
                pass
            
            # *text_SATI_short_low_clarification* updates
            
            # if text_SATI_short_low_clarification is starting this frame...
            if text_SATI_short_low_clarification.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_low_clarification.frameNStart = frameN  # exact frame index
                text_SATI_short_low_clarification.tStart = t  # local t and not account for scr refresh
                text_SATI_short_low_clarification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_low_clarification, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_low_clarification.status = STARTED
                text_SATI_short_low_clarification.setAutoDraw(True)
            
            # if text_SATI_short_low_clarification is active this frame...
            if text_SATI_short_low_clarification.status == STARTED:
                # update params
                pass
            
            # *slider_SATI_short_low* updates
            
            # if slider_SATI_short_low is starting this frame...
            if slider_SATI_short_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_SATI_short_low.frameNStart = frameN  # exact frame index
                slider_SATI_short_low.tStart = t  # local t and not account for scr refresh
                slider_SATI_short_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SATI_short_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SATI_short_low.started')
                # update status
                slider_SATI_short_low.status = STARTED
                slider_SATI_short_low.setAutoDraw(True)
            
            # if slider_SATI_short_low is active this frame...
            if slider_SATI_short_low.status == STARTED:
                # update params
                pass
            # *mouse_SATI_short_low* updates
            
            # if mouse_SATI_short_low is starting this frame...
            if mouse_SATI_short_low.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_SATI_short_low.frameNStart = frameN  # exact frame index
                mouse_SATI_short_low.tStart = t  # local t and not account for scr refresh
                mouse_SATI_short_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_SATI_short_low, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_SATI_short_low.status = STARTED
                mouse_SATI_short_low.mouseClock.reset()
                prevButtonState = mouse_SATI_short_low.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_SATI_short_low
            mouse_pos = mouse_SATI_short_low.getPos()
            
            x_min = slider_SATI_short_low.pos[0] - slider_SATI_short_low.size[0] / 2
            x_max = slider_SATI_short_low.pos[0] + slider_SATI_short_low.size[0] / 2
            y_min = slider_SATI_short_low.pos[1] - 0.1
            y_max = slider_SATI_short_low.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_SATI_short_low.markerPos = slider_SATI_short_low._posToRating(mouse_pos)
            
            buttonState = mouse_SATI_short_low.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_SATI_short_low.rating = slider_SATI_short_low.markerPos
                    responseSelected = True
            
            # Wait until the mouse has been released for 0.8 s
            if responseSelected:
                if buttonState == 0:
                    if releaseStartTime is None:
                        releaseStartTime = t
                    elif t - releaseStartTime >= releaseWaitTime:
                        continueRoutine = False
                else:
                    releaseStartTime = None
            
            prevButtonState = buttonState
            
            # *prog_short_low* updates
            
            # if prog_short_low is starting this frame...
            if prog_short_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_short_low.frameNStart = frameN  # exact frame index
                prog_short_low.tStart = t  # local t and not account for scr refresh
                prog_short_low.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_short_low, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_short_low.started')
                # update status
                prog_short_low.status = STARTED
                prog_short_low.setAutoDraw(True)
            
            # if prog_short_low is active this frame...
            if prog_short_low.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=SATI_short_low,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                SATI_short_low.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if SATI_short_low.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in SATI_short_low.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SATI_short_low" ---
        for thisComponent in SATI_short_low.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SATI_short_low
        SATI_short_low.tStop = globalClock.getTime(format='float')
        SATI_short_low.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SATI_short_low.stopped', SATI_short_low.tStop)
        # store data for SATI_low (TrialHandler)
        # Run 'End Routine' code from code_SATI_short_low
        if slider_SATI_short_low.markerPos is not None:
            thisExp.addData('answ_SATI_short_low', slider_SATI_short_low.markerPos)
        else:
            thisExp.addData('answ_SATI_short_low', 'NA')
        # the Routine "SATI_short_low" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisSATI_low as finished
        if hasattr(thisSATI_low, 'status'):
            thisSATI_low.status = FINISHED
        # if awaiting a pause, pause now
        if SATI_low.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            SATI_low.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'SATI_low'
    SATI_low.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "how_you_feel_low" ---
    # create an object to store info about Routine how_you_feel_low
    how_you_feel_low = data.Routine(
        name='how_you_feel_low',
        components=[pleasant_low, unpleasant_low, text_how_you_feel_low],
    )
    how_you_feel_low.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset pleasant_low to account for continued clicks & clear times on/off
    pleasant_low.reset()
    # reset unpleasant_low to account for continued clicks & clear times on/off
    unpleasant_low.reset()
    # store start times for how_you_feel_low
    how_you_feel_low.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    how_you_feel_low.tStart = globalClock.getTime(format='float')
    how_you_feel_low.status = STARTED
    thisExp.addData('how_you_feel_low.started', how_you_feel_low.tStart)
    how_you_feel_low.maxDuration = None
    # keep track of which components have finished
    how_you_feel_lowComponents = how_you_feel_low.components
    for thisComponent in how_you_feel_low.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "how_you_feel_low" ---
    thisExp.currentRoutine = how_you_feel_low
    how_you_feel_low.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pleasant_low* updates
        
        # if pleasant_low is starting this frame...
        if pleasant_low.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            pleasant_low.frameNStart = frameN  # exact frame index
            pleasant_low.tStart = t  # local t and not account for scr refresh
            pleasant_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pleasant_low, 'tStartRefresh')  # time at next scr refresh
            # update status
            pleasant_low.status = STARTED
            win.callOnFlip(pleasant_low.buttonClock.reset)
            pleasant_low.setAutoDraw(True)
        
        # if pleasant_low is active this frame...
        if pleasant_low.status == STARTED:
            # update params
            pass
            # check whether pleasant_low has been pressed
            if pleasant_low.isClicked:
                if not pleasant_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    pleasant_low.timesOn.append(pleasant_low.buttonClock.getTime())
                    pleasant_low.timesOff.append(pleasant_low.buttonClock.getTime())
                elif len(pleasant_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    pleasant_low.timesOff[-1] = pleasant_low.buttonClock.getTime()
                if not pleasant_low.wasClicked:
                    # end routine when pleasant_low is clicked
                    continueRoutine = False
                if not pleasant_low.wasClicked:
                    # run callback code when pleasant_low is clicked
                    pass
        # take note of whether pleasant_low was clicked, so that next frame we know if clicks are new
        pleasant_low.wasClicked = pleasant_low.isClicked and pleasant_low.status == STARTED
        # *unpleasant_low* updates
        
        # if unpleasant_low is starting this frame...
        if unpleasant_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            unpleasant_low.frameNStart = frameN  # exact frame index
            unpleasant_low.tStart = t  # local t and not account for scr refresh
            unpleasant_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(unpleasant_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'unpleasant_low.started')
            # update status
            unpleasant_low.status = STARTED
            win.callOnFlip(unpleasant_low.buttonClock.reset)
            unpleasant_low.setAutoDraw(True)
        
        # if unpleasant_low is active this frame...
        if unpleasant_low.status == STARTED:
            # update params
            pass
            # check whether unpleasant_low has been pressed
            if unpleasant_low.isClicked:
                if not unpleasant_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    unpleasant_low.timesOn.append(unpleasant_low.buttonClock.getTime())
                    unpleasant_low.timesOff.append(unpleasant_low.buttonClock.getTime())
                elif len(unpleasant_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    unpleasant_low.timesOff[-1] = unpleasant_low.buttonClock.getTime()
                if not unpleasant_low.wasClicked:
                    # end routine when unpleasant_low is clicked
                    continueRoutine = False
                if not unpleasant_low.wasClicked:
                    # run callback code when unpleasant_low is clicked
                    pass
        # take note of whether unpleasant_low was clicked, so that next frame we know if clicks are new
        unpleasant_low.wasClicked = unpleasant_low.isClicked and unpleasant_low.status == STARTED
        
        # *text_how_you_feel_low* updates
        
        # if text_how_you_feel_low is starting this frame...
        if text_how_you_feel_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_how_you_feel_low.frameNStart = frameN  # exact frame index
            text_how_you_feel_low.tStart = t  # local t and not account for scr refresh
            text_how_you_feel_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_how_you_feel_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_how_you_feel_low.started')
            # update status
            text_how_you_feel_low.status = STARTED
            text_how_you_feel_low.setAutoDraw(True)
        
        # if text_how_you_feel_low is active this frame...
        if text_how_you_feel_low.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=how_you_feel_low,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            how_you_feel_low.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if how_you_feel_low.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in how_you_feel_low.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "how_you_feel_low" ---
    for thisComponent in how_you_feel_low.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for how_you_feel_low
    how_you_feel_low.tStop = globalClock.getTime(format='float')
    how_you_feel_low.tStopRefresh = tThisFlipGlobal
    thisExp.addData('how_you_feel_low.stopped', how_you_feel_low.tStop)
    thisExp.addData('pleasant_low.numClicks', pleasant_low.numClicks)
    if pleasant_low.numClicks:
       thisExp.addData('pleasant_low.timesOn', pleasant_low.timesOn)
       thisExp.addData('pleasant_low.timesOff', pleasant_low.timesOff)
    else:
       thisExp.addData('pleasant_low.timesOn', "")
       thisExp.addData('pleasant_low.timesOff', "")
    thisExp.addData('unpleasant_low.numClicks', unpleasant_low.numClicks)
    if unpleasant_low.numClicks:
       thisExp.addData('unpleasant_low.timesOn', unpleasant_low.timesOn)
       thisExp.addData('unpleasant_low.timesOff', unpleasant_low.timesOff)
    else:
       thisExp.addData('unpleasant_low.timesOn', "")
       thisExp.addData('unpleasant_low.timesOff', "")
    # Run 'End Routine' code from code_how_you_feel_low
    if pleasant_low.isClicked:
        valence_answer = "pleasant"
    elif unpleasant_low.isClicked:
        valence_answer = "unpleasant"
    else:
        valence_answer = "no_response"
    
    thisExp.addData('answ_valence', valence_answer)
    thisExp.nextEntry()
    # the Routine "how_you_feel_low" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_low_choice" ---
    # create an object to store info about Routine pleasant_low_choice
    pleasant_low_choice = data.Routine(
        name='pleasant_low_choice',
        components=[excited_low, relaxed_low, text_pleasant_low],
    )
    pleasant_low_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset excited_low to account for continued clicks & clear times on/off
    excited_low.reset()
    # reset relaxed_low to account for continued clicks & clear times on/off
    relaxed_low.reset()
    # Run 'Begin Routine' code from code_pleasant_low
    # Only show this routine if "pleasant" was selected
    if valence_answer != "pleasant":
        continueRoutine = False
    # store start times for pleasant_low_choice
    pleasant_low_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pleasant_low_choice.tStart = globalClock.getTime(format='float')
    pleasant_low_choice.status = STARTED
    thisExp.addData('pleasant_low_choice.started', pleasant_low_choice.tStart)
    pleasant_low_choice.maxDuration = None
    # keep track of which components have finished
    pleasant_low_choiceComponents = pleasant_low_choice.components
    for thisComponent in pleasant_low_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_low_choice" ---
    thisExp.currentRoutine = pleasant_low_choice
    pleasant_low_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *excited_low* updates
        
        # if excited_low is starting this frame...
        if excited_low.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            excited_low.frameNStart = frameN  # exact frame index
            excited_low.tStart = t  # local t and not account for scr refresh
            excited_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(excited_low, 'tStartRefresh')  # time at next scr refresh
            # update status
            excited_low.status = STARTED
            win.callOnFlip(excited_low.buttonClock.reset)
            excited_low.setAutoDraw(True)
        
        # if excited_low is active this frame...
        if excited_low.status == STARTED:
            # update params
            pass
            # check whether excited_low has been pressed
            if excited_low.isClicked:
                if not excited_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    excited_low.timesOn.append(excited_low.buttonClock.getTime())
                    excited_low.timesOff.append(excited_low.buttonClock.getTime())
                elif len(excited_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    excited_low.timesOff[-1] = excited_low.buttonClock.getTime()
                if not excited_low.wasClicked:
                    # end routine when excited_low is clicked
                    continueRoutine = False
                if not excited_low.wasClicked:
                    # run callback code when excited_low is clicked
                    pass
        # take note of whether excited_low was clicked, so that next frame we know if clicks are new
        excited_low.wasClicked = excited_low.isClicked and excited_low.status == STARTED
        # *relaxed_low* updates
        
        # if relaxed_low is starting this frame...
        if relaxed_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            relaxed_low.frameNStart = frameN  # exact frame index
            relaxed_low.tStart = t  # local t and not account for scr refresh
            relaxed_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(relaxed_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'relaxed_low.started')
            # update status
            relaxed_low.status = STARTED
            win.callOnFlip(relaxed_low.buttonClock.reset)
            relaxed_low.setAutoDraw(True)
        
        # if relaxed_low is active this frame...
        if relaxed_low.status == STARTED:
            # update params
            pass
            # check whether relaxed_low has been pressed
            if relaxed_low.isClicked:
                if not relaxed_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    relaxed_low.timesOn.append(relaxed_low.buttonClock.getTime())
                    relaxed_low.timesOff.append(relaxed_low.buttonClock.getTime())
                elif len(relaxed_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    relaxed_low.timesOff[-1] = relaxed_low.buttonClock.getTime()
                if not relaxed_low.wasClicked:
                    # end routine when relaxed_low is clicked
                    continueRoutine = False
                if not relaxed_low.wasClicked:
                    # run callback code when relaxed_low is clicked
                    pass
        # take note of whether relaxed_low was clicked, so that next frame we know if clicks are new
        relaxed_low.wasClicked = relaxed_low.isClicked and relaxed_low.status == STARTED
        
        # *text_pleasant_low* updates
        
        # if text_pleasant_low is starting this frame...
        if text_pleasant_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pleasant_low.frameNStart = frameN  # exact frame index
            text_pleasant_low.tStart = t  # local t and not account for scr refresh
            text_pleasant_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pleasant_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pleasant_low.started')
            # update status
            text_pleasant_low.status = STARTED
            text_pleasant_low.setAutoDraw(True)
        
        # if text_pleasant_low is active this frame...
        if text_pleasant_low.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=pleasant_low_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            pleasant_low_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if pleasant_low_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in pleasant_low_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_low_choice" ---
    for thisComponent in pleasant_low_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pleasant_low_choice
    pleasant_low_choice.tStop = globalClock.getTime(format='float')
    pleasant_low_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pleasant_low_choice.stopped', pleasant_low_choice.tStop)
    thisExp.addData('excited_low.numClicks', excited_low.numClicks)
    if excited_low.numClicks:
       thisExp.addData('excited_low.timesOn', excited_low.timesOn)
       thisExp.addData('excited_low.timesOff', excited_low.timesOff)
    else:
       thisExp.addData('excited_low.timesOn', "")
       thisExp.addData('excited_low.timesOff', "")
    thisExp.addData('relaxed_low.numClicks', relaxed_low.numClicks)
    if relaxed_low.numClicks:
       thisExp.addData('relaxed_low.timesOn', relaxed_low.timesOn)
       thisExp.addData('relaxed_low.timesOff', relaxed_low.timesOff)
    else:
       thisExp.addData('relaxed_low.timesOn', "")
       thisExp.addData('relaxed_low.timesOff', "")
    # Run 'End Routine' code from code_pleasant_low
    if excited_low.isClicked:
        pleasant_answer = "excited"
    elif relaxed_low.isClicked:
        pleasant_answer = "relaxed"
    else:
        pleasant_answer = "no_response"
    
    thisExp.addData('answ_pleasant', pleasant_answer)
    thisExp.nextEntry()
    # the Routine "pleasant_low_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "unpleasant_low_choice" ---
    # create an object to store info about Routine unpleasant_low_choice
    unpleasant_low_choice = data.Routine(
        name='unpleasant_low_choice',
        components=[stressed_low, bored_low, text_unpleasant_low],
    )
    unpleasant_low_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset stressed_low to account for continued clicks & clear times on/off
    stressed_low.reset()
    # reset bored_low to account for continued clicks & clear times on/off
    bored_low.reset()
    # Run 'Begin Routine' code from code_unpleasant_low
    # Only show this routine if "unpleasant" was selected
    if valence_answer != "unpleasant":
        continueRoutine = False
    # store start times for unpleasant_low_choice
    unpleasant_low_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    unpleasant_low_choice.tStart = globalClock.getTime(format='float')
    unpleasant_low_choice.status = STARTED
    thisExp.addData('unpleasant_low_choice.started', unpleasant_low_choice.tStart)
    unpleasant_low_choice.maxDuration = None
    # keep track of which components have finished
    unpleasant_low_choiceComponents = unpleasant_low_choice.components
    for thisComponent in unpleasant_low_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "unpleasant_low_choice" ---
    thisExp.currentRoutine = unpleasant_low_choice
    unpleasant_low_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *stressed_low* updates
        
        # if stressed_low is starting this frame...
        if stressed_low.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            stressed_low.frameNStart = frameN  # exact frame index
            stressed_low.tStart = t  # local t and not account for scr refresh
            stressed_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stressed_low, 'tStartRefresh')  # time at next scr refresh
            # update status
            stressed_low.status = STARTED
            win.callOnFlip(stressed_low.buttonClock.reset)
            stressed_low.setAutoDraw(True)
        
        # if stressed_low is active this frame...
        if stressed_low.status == STARTED:
            # update params
            pass
            # check whether stressed_low has been pressed
            if stressed_low.isClicked:
                if not stressed_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    stressed_low.timesOn.append(stressed_low.buttonClock.getTime())
                    stressed_low.timesOff.append(stressed_low.buttonClock.getTime())
                elif len(stressed_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    stressed_low.timesOff[-1] = stressed_low.buttonClock.getTime()
                if not stressed_low.wasClicked:
                    # end routine when stressed_low is clicked
                    continueRoutine = False
                if not stressed_low.wasClicked:
                    # run callback code when stressed_low is clicked
                    pass
        # take note of whether stressed_low was clicked, so that next frame we know if clicks are new
        stressed_low.wasClicked = stressed_low.isClicked and stressed_low.status == STARTED
        # *bored_low* updates
        
        # if bored_low is starting this frame...
        if bored_low.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            bored_low.frameNStart = frameN  # exact frame index
            bored_low.tStart = t  # local t and not account for scr refresh
            bored_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bored_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bored_low.started')
            # update status
            bored_low.status = STARTED
            win.callOnFlip(bored_low.buttonClock.reset)
            bored_low.setAutoDraw(True)
        
        # if bored_low is active this frame...
        if bored_low.status == STARTED:
            # update params
            pass
            # check whether bored_low has been pressed
            if bored_low.isClicked:
                if not bored_low.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    bored_low.timesOn.append(bored_low.buttonClock.getTime())
                    bored_low.timesOff.append(bored_low.buttonClock.getTime())
                elif len(bored_low.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    bored_low.timesOff[-1] = bored_low.buttonClock.getTime()
                if not bored_low.wasClicked:
                    # end routine when bored_low is clicked
                    continueRoutine = False
                if not bored_low.wasClicked:
                    # run callback code when bored_low is clicked
                    pass
        # take note of whether bored_low was clicked, so that next frame we know if clicks are new
        bored_low.wasClicked = bored_low.isClicked and bored_low.status == STARTED
        
        # *text_unpleasant_low* updates
        
        # if text_unpleasant_low is starting this frame...
        if text_unpleasant_low.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_unpleasant_low.frameNStart = frameN  # exact frame index
            text_unpleasant_low.tStart = t  # local t and not account for scr refresh
            text_unpleasant_low.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_unpleasant_low, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_unpleasant_low.started')
            # update status
            text_unpleasant_low.status = STARTED
            text_unpleasant_low.setAutoDraw(True)
        
        # if text_unpleasant_low is active this frame...
        if text_unpleasant_low.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=unpleasant_low_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            unpleasant_low_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if unpleasant_low_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in unpleasant_low_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "unpleasant_low_choice" ---
    for thisComponent in unpleasant_low_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for unpleasant_low_choice
    unpleasant_low_choice.tStop = globalClock.getTime(format='float')
    unpleasant_low_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('unpleasant_low_choice.stopped', unpleasant_low_choice.tStop)
    thisExp.addData('stressed_low.numClicks', stressed_low.numClicks)
    if stressed_low.numClicks:
       thisExp.addData('stressed_low.timesOn', stressed_low.timesOn)
       thisExp.addData('stressed_low.timesOff', stressed_low.timesOff)
    else:
       thisExp.addData('stressed_low.timesOn', "")
       thisExp.addData('stressed_low.timesOff', "")
    thisExp.addData('bored_low.numClicks', bored_low.numClicks)
    if bored_low.numClicks:
       thisExp.addData('bored_low.timesOn', bored_low.timesOn)
       thisExp.addData('bored_low.timesOff', bored_low.timesOff)
    else:
       thisExp.addData('bored_low.timesOn', "")
       thisExp.addData('bored_low.timesOff', "")
    # Run 'End Routine' code from code_unpleasant_low
    if stressed_low.isClicked:
        unpleasant_answer = "stressed"
    elif bored_low.isClicked:
        unpleasant_answer = "bored"
    else:
        unpleasant_answer = "no_response"
    
    thisExp.addData('answ_unpleasant', unpleasant_answer)
    thisExp.nextEntry()
    # the Routine "unpleasant_low_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "high_reliability_start" ---
    # create an object to store info about Routine high_reliability_start
    high_reliability_start = data.Routine(
        name='high_reliability_start',
        components=[text_high_reliability_start, key_resp_high_reliability_start, text_space_high_reliability_start],
    )
    high_reliability_start.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_high_reliability_start
    key_resp_high_reliability_start.keys = []
    key_resp_high_reliability_start.rt = []
    _key_resp_high_reliability_start_allKeys = []
    # store start times for high_reliability_start
    high_reliability_start.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    high_reliability_start.tStart = globalClock.getTime(format='float')
    high_reliability_start.status = STARTED
    thisExp.addData('high_reliability_start.started', high_reliability_start.tStart)
    high_reliability_start.maxDuration = None
    # keep track of which components have finished
    high_reliability_startComponents = high_reliability_start.components
    for thisComponent in high_reliability_start.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "high_reliability_start" ---
    thisExp.currentRoutine = high_reliability_start
    high_reliability_start.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_high_reliability_start* updates
        
        # if text_high_reliability_start is starting this frame...
        if text_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_high_reliability_start.frameNStart = frameN  # exact frame index
            text_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_high_reliability_start.status = STARTED
            text_high_reliability_start.setAutoDraw(True)
        
        # if text_high_reliability_start is active this frame...
        if text_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # *key_resp_high_reliability_start* updates
        
        # if key_resp_high_reliability_start is starting this frame...
        if key_resp_high_reliability_start.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_high_reliability_start.frameNStart = frameN  # exact frame index
            key_resp_high_reliability_start.tStart = t  # local t and not account for scr refresh
            key_resp_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_high_reliability_start.status = STARTED
            # keyboard checking is just starting
            key_resp_high_reliability_start.clock.reset()  # now t=0
        if key_resp_high_reliability_start.status == STARTED:
            theseKeys = key_resp_high_reliability_start.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_high_reliability_start_allKeys.extend(theseKeys)
            if len(_key_resp_high_reliability_start_allKeys):
                key_resp_high_reliability_start.keys = _key_resp_high_reliability_start_allKeys[-1].name  # just the last key pressed
                key_resp_high_reliability_start.rt = _key_resp_high_reliability_start_allKeys[-1].rt
                key_resp_high_reliability_start.duration = _key_resp_high_reliability_start_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_high_reliability_start* updates
        
        # if text_space_high_reliability_start is starting this frame...
        if text_space_high_reliability_start.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_high_reliability_start.frameNStart = frameN  # exact frame index
            text_space_high_reliability_start.tStart = t  # local t and not account for scr refresh
            text_space_high_reliability_start.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_high_reliability_start, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_high_reliability_start.status = STARTED
            text_space_high_reliability_start.setAutoDraw(True)
        
        # if text_space_high_reliability_start is active this frame...
        if text_space_high_reliability_start.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=high_reliability_start,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            high_reliability_start.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if high_reliability_start.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in high_reliability_start.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "high_reliability_start" ---
    for thisComponent in high_reliability_start.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for high_reliability_start
    high_reliability_start.tStop = globalClock.getTime(format='float')
    high_reliability_start.tStopRefresh = tThisFlipGlobal
    thisExp.addData('high_reliability_start.stopped', high_reliability_start.tStop)
    thisExp.nextEntry()
    # the Routine "high_reliability_start" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    high_3 = data.TrialHandler2(
        name='high_3',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('trial_100_3_ecological.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(high_3)  # add the loop to the experiment
    thisHigh_3 = high_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisHigh_3.rgb)
    if thisHigh_3 != None:
        for paramName in thisHigh_3:
            globals()[paramName] = thisHigh_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisHigh_3 in high_3:
        high_3.status = STARTED
        if hasattr(thisHigh_3, 'status'):
            thisHigh_3.status = STARTED
        currentLoop = high_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisHigh_3.rgb)
        if thisHigh_3 != None:
            for paramName in thisHigh_3:
                globals()[paramName] = thisHigh_3[paramName]
        
        # --- Prepare to start Routine "time_high" ---
        # create an object to store info about Routine time_high
        time_high = data.Routine(
            name='time_high',
            components=[],
        )
        time_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for time_high
        time_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        time_high.tStart = globalClock.getTime(format='float')
        time_high.status = STARTED
        time_high.maxDuration = 1
        # keep track of which components have finished
        time_highComponents = time_high.components
        for thisComponent in time_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "time_high" ---
        thisExp.currentRoutine = time_high
        time_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_3, 'status') and thisHigh_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > time_high.maxDuration-frameTolerance:
                time_high.maxDurationReached = True
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=time_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                time_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if time_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in time_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "time_high" ---
        for thisComponent in time_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for time_high
        time_high.tStop = globalClock.getTime(format='float')
        time_high.tStopRefresh = tThisFlipGlobal
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if time_high.maxDurationReached:
            routineTimer.addTime(-time_high.maxDuration)
        elif time_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "fixation_cross" ---
        # create an object to store info about Routine fixation_cross
        fixation_cross = data.Routine(
            name='fixation_cross',
            components=[fixation_cross_look],
        )
        fixation_cross.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_START | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # store start times for fixation_cross
        fixation_cross.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_cross.tStart = globalClock.getTime(format='float')
        fixation_cross.status = STARTED
        thisExp.addData('fixation_cross.started', fixation_cross.tStart)
        fixation_cross.maxDuration = None
        # keep track of which components have finished
        fixation_crossComponents = fixation_cross.components
        for thisComponent in fixation_cross.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_cross" ---
        thisExp.currentRoutine = fixation_cross
        fixation_cross.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_3, 'status') and thisHigh_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation_cross_look* updates
            
            # if fixation_cross_look is starting this frame...
            if fixation_cross_look.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation_cross_look.frameNStart = frameN  # exact frame index
                fixation_cross_look.tStart = t  # local t and not account for scr refresh
                fixation_cross_look.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_cross_look, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_cross_look.started')
                # update status
                fixation_cross_look.status = STARTED
                fixation_cross_look.setAutoDraw(True)
            
            # if fixation_cross_look is active this frame...
            if fixation_cross_look.status == STARTED:
                # update params
                pass
            
            # if fixation_cross_look is stopping this frame...
            if fixation_cross_look.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_cross_look.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_cross_look.tStop = t  # not accounting for scr refresh
                    fixation_cross_look.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_cross_look.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_cross_look.stopped')
                    # update status
                    fixation_cross_look.status = FINISHED
                    fixation_cross_look.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=fixation_cross,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                fixation_cross.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if fixation_cross.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in fixation_cross.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_cross" ---
        for thisComponent in fixation_cross.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_cross
        fixation_cross.tStop = globalClock.getTime(format='float')
        fixation_cross.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_cross.stopped', fixation_cross.tStop)
        # Run 'End Routine' code from code_fixation_cross_look
        marker = f"FIXATION_CROSS_END | BLOCK_{block} | Trial={block_trial}"
        outlet.push_sample([marker])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_cross.maxDurationReached:
            routineTimer.addTime(-fixation_cross.maxDuration)
        elif fixation_cross.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        
        # --- Prepare to start Routine "trial_high" ---
        # create an object to store info about Routine trial_high
        trial_high = data.Routine(
            name='trial_high',
            components=[trial_high_aircraft_1, trial_high_aircraft_2, change_high, maintain_high, support_high],
        )
        trial_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        trial_high_aircraft_1.setText(a1)
        trial_high_aircraft_2.setText(a2)
        # reset change_high to account for continued clicks & clear times on/off
        change_high.reset()
        # reset maintain_high to account for continued clicks & clear times on/off
        maintain_high.reset()
        support_high.setPos((value, -0.3))
        # Run 'Begin Routine' code from code_high
        # Start time
        t_start = globalClock.getTime()
        
        # Interpret automated recommendation
        if value < 0:
            recommendation = "change"
        else:
            recommendation = "maintain"
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Aircraft_1={a1} | "
            f"Aircraft_2={a2} | "
            f"Loss_of_separation={conflict} | "
            f"Automation_recommendation={recommendation} | "
            f"Automation_error={automation_error}"
        )
        outlet.push_sample([marker_info_low])
        
        # Start marker
        marker_start_low = f"START | T{block_trial} | B{block}"
        outlet.push_sample([marker_start_low])
        # store start times for trial_high
        trial_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_high.tStart = globalClock.getTime(format='float')
        trial_high.status = STARTED
        thisExp.addData('trial_high.started', trial_high.tStart)
        trial_high.maxDuration = 3
        # keep track of which components have finished
        trial_highComponents = trial_high.components
        for thisComponent in trial_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_high" ---
        thisExp.currentRoutine = trial_high
        trial_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_3, 'status') and thisHigh_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # is it time to end the Routine? (based on local clock)
            if tThisFlip > trial_high.maxDuration-frameTolerance:
                trial_high.maxDurationReached = True
                continueRoutine = False
            
            # *trial_high_aircraft_1* updates
            
            # if trial_high_aircraft_1 is starting this frame...
            if trial_high_aircraft_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_1.frameNStart = frameN  # exact frame index
                trial_high_aircraft_1.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'trial_high_aircraft_1.started')
                # update status
                trial_high_aircraft_1.status = STARTED
                trial_high_aircraft_1.setAutoDraw(True)
            
            # if trial_high_aircraft_1 is active this frame...
            if trial_high_aircraft_1.status == STARTED:
                # update params
                pass
            
            # *trial_high_aircraft_2* updates
            
            # if trial_high_aircraft_2 is starting this frame...
            if trial_high_aircraft_2.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                trial_high_aircraft_2.frameNStart = frameN  # exact frame index
                trial_high_aircraft_2.tStart = t  # local t and not account for scr refresh
                trial_high_aircraft_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(trial_high_aircraft_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                trial_high_aircraft_2.status = STARTED
                trial_high_aircraft_2.setAutoDraw(True)
            
            # if trial_high_aircraft_2 is active this frame...
            if trial_high_aircraft_2.status == STARTED:
                # update params
                pass
            # *change_high* updates
            
            # if change_high is starting this frame...
            if change_high.status == NOT_STARTED and t >= 0-frameTolerance:
                # keep track of start time/frame for later
                change_high.frameNStart = frameN  # exact frame index
                change_high.tStart = t  # local t and not account for scr refresh
                change_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(change_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                change_high.status = STARTED
                win.callOnFlip(change_high.buttonClock.reset)
                change_high.setAutoDraw(True)
            
            # if change_high is active this frame...
            if change_high.status == STARTED:
                # update params
                pass
                # check whether change_high has been pressed
                if change_high.isClicked:
                    if not change_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        change_high.timesOn.append(change_high.buttonClock.getTime())
                        change_high.timesOff.append(change_high.buttonClock.getTime())
                    elif len(change_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        change_high.timesOff[-1] = change_high.buttonClock.getTime()
                    if not change_high.wasClicked:
                        # end routine when change_high is clicked
                        continueRoutine = False
                    if not change_high.wasClicked:
                        # run callback code when change_high is clicked
                        pass
            # take note of whether change_high was clicked, so that next frame we know if clicks are new
            change_high.wasClicked = change_high.isClicked and change_high.status == STARTED
            # *maintain_high* updates
            
            # if maintain_high is starting this frame...
            if maintain_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                maintain_high.frameNStart = frameN  # exact frame index
                maintain_high.tStart = t  # local t and not account for scr refresh
                maintain_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(maintain_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'maintain_high.started')
                # update status
                maintain_high.status = STARTED
                win.callOnFlip(maintain_high.buttonClock.reset)
                maintain_high.setAutoDraw(True)
            
            # if maintain_high is active this frame...
            if maintain_high.status == STARTED:
                # update params
                pass
                # check whether maintain_high has been pressed
                if maintain_high.isClicked:
                    if not maintain_high.wasClicked:
                        # if this is a new click, store time of first click and clicked until
                        maintain_high.timesOn.append(maintain_high.buttonClock.getTime())
                        maintain_high.timesOff.append(maintain_high.buttonClock.getTime())
                    elif len(maintain_high.timesOff):
                        # if click is continuing from last frame, update time of clicked until
                        maintain_high.timesOff[-1] = maintain_high.buttonClock.getTime()
                    if not maintain_high.wasClicked:
                        # end routine when maintain_high is clicked
                        continueRoutine = False
                    if not maintain_high.wasClicked:
                        # run callback code when maintain_high is clicked
                        pass
            # take note of whether maintain_high was clicked, so that next frame we know if clicks are new
            maintain_high.wasClicked = maintain_high.isClicked and maintain_high.status == STARTED
            
            # *support_high* updates
            
            # if support_high is starting this frame...
            if support_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                support_high.frameNStart = frameN  # exact frame index
                support_high.tStart = t  # local t and not account for scr refresh
                support_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(support_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'support_high.started')
                # update status
                support_high.status = STARTED
                support_high.setAutoDraw(True)
            
            # if support_high is active this frame...
            if support_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                trial_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if trial_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in trial_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_high" ---
        for thisComponent in trial_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_high
        trial_high.tStop = globalClock.getTime(format='float')
        trial_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_high.stopped', trial_high.tStop)
        high_3.addData('change_high.numClicks', change_high.numClicks)
        if change_high.numClicks:
           high_3.addData('change_high.timesOn', change_high.timesOn)
           high_3.addData('change_high.timesOff', change_high.timesOff)
        else:
           high_3.addData('change_high.timesOn', "")
           high_3.addData('change_high.timesOff', "")
        high_3.addData('maintain_high.numClicks', maintain_high.numClicks)
        if maintain_high.numClicks:
           high_3.addData('maintain_high.timesOn', maintain_high.timesOn)
           high_3.addData('maintain_high.timesOff', maintain_high.timesOff)
        else:
           high_3.addData('maintain_high.timesOn', "")
           high_3.addData('maintain_high.timesOff', "")
        # Run 'End Routine' code from code_high
        # Detect which button was pressed
        if change_high.isClicked:
            response = "change"
        elif maintain_high.isClicked:
            response = "maintain"
        else:
            response = "no_response"
        
        # Normalize Excel value
        con = str(conflict).strip().lower()
        
        # Evaluate correctness
        if con == "yes" and response == "change":
            result = "Correct action"
            correct = 1
        elif con == "no" and response == "maintain":
            result = "Correct action"
            correct = 1
        elif con == "yes" and response == "maintain":
            result = "Incorrect action"
            correct = 0
        elif con == "no" and response == "change":
            result = "Incorrect action"
            correct = 0
        else:
            result = "Time out"
            correct = 0
        
        # Save data
        thisExp.addData('answer', response)
        thisExp.addData('correct', correct)
        
        # Marker with trial information
        marker_info_low = (
            f"Trial={block_trial} | "
            f"Block={block} | "
            f"Response={response} | "
            f"Correct={correct}"
        )
        outlet.push_sample([marker_info_low])
        
        # Trial end marker
        marker_end_low = f"END | T{block_trial} | B{block}"
        outlet.push_sample([marker_end_low])
        # the Routine "trial_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "answer_high" ---
        # create an object to store info about Routine answer_high
        answer_high = data.Routine(
            name='answer_high',
            components=[text_answer_high],
        )
        answer_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_answer_high.setText(result)
        # store start times for answer_high
        answer_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        answer_high.tStart = globalClock.getTime(format='float')
        answer_high.status = STARTED
        thisExp.addData('answer_high.started', answer_high.tStart)
        answer_high.maxDuration = None
        # keep track of which components have finished
        answer_highComponents = answer_high.components
        for thisComponent in answer_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "answer_high" ---
        thisExp.currentRoutine = answer_high
        answer_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_3, 'status') and thisHigh_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_answer_high* updates
            
            # if text_answer_high is starting this frame...
            if text_answer_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_answer_high.frameNStart = frameN  # exact frame index
                text_answer_high.tStart = t  # local t and not account for scr refresh
                text_answer_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_answer_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_answer_high.started')
                # update status
                text_answer_high.status = STARTED
                text_answer_high.setAutoDraw(True)
            
            # if text_answer_high is active this frame...
            if text_answer_high.status == STARTED:
                # update params
                pass
            
            # if text_answer_high is stopping this frame...
            if text_answer_high.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_answer_high.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    text_answer_high.tStop = t  # not accounting for scr refresh
                    text_answer_high.tStopRefresh = tThisFlipGlobal  # on global time
                    text_answer_high.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_answer_high.stopped')
                    # update status
                    text_answer_high.status = FINISHED
                    text_answer_high.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=answer_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                answer_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if answer_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in answer_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "answer_high" ---
        for thisComponent in answer_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for answer_high
        answer_high.tStop = globalClock.getTime(format='float')
        answer_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('answer_high.stopped', answer_high.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if answer_high.maxDurationReached:
            routineTimer.addTime(-answer_high.maxDuration)
        elif answer_high.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "question_high" ---
        # create an object to store info about Routine question_high
        question_high = data.Routine(
            name='question_high',
            components=[text_high_question, slider_high_question, mouse_high_question],
        )
        question_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_high_question.setText('How much do you trust the automation?')
        slider_high_question.reset()
        # setup some python lists for storing info about the mouse_high_question
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_high_question
        # Reset mouse clicks
        mouse_high_question.clickReset()
        
        # Previous button state
        prevButtonState = mouse_high_question.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        # store start times for question_high
        question_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        question_high.tStart = globalClock.getTime(format='float')
        question_high.status = STARTED
        thisExp.addData('question_high.started', question_high.tStart)
        question_high.maxDuration = None
        # keep track of which components have finished
        question_highComponents = question_high.components
        for thisComponent in question_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "question_high" ---
        thisExp.currentRoutine = question_high
        question_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisHigh_3, 'status') and thisHigh_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_high_question* updates
            
            # if text_high_question is starting this frame...
            if text_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_high_question.frameNStart = frameN  # exact frame index
                text_high_question.tStart = t  # local t and not account for scr refresh
                text_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_high_question.status = STARTED
                text_high_question.setAutoDraw(True)
            
            # if text_high_question is active this frame...
            if text_high_question.status == STARTED:
                # update params
                pass
            
            # *slider_high_question* updates
            
            # if slider_high_question is starting this frame...
            if slider_high_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_high_question.frameNStart = frameN  # exact frame index
                slider_high_question.tStart = t  # local t and not account for scr refresh
                slider_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_high_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_high_question.started')
                # update status
                slider_high_question.status = STARTED
                slider_high_question.setAutoDraw(True)
            
            # if slider_high_question is active this frame...
            if slider_high_question.status == STARTED:
                # update params
                pass
            
            # Check slider_high_question for response to end Routine
            if slider_high_question.getRating() is not None and slider_high_question.status == STARTED:
                continueRoutine = False
            # *mouse_high_question* updates
            
            # if mouse_high_question is starting this frame...
            if mouse_high_question.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_high_question.frameNStart = frameN  # exact frame index
                mouse_high_question.tStart = t  # local t and not account for scr refresh
                mouse_high_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_high_question, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_high_question.status = STARTED
                mouse_high_question.mouseClock.reset()
                prevButtonState = mouse_high_question.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_high_question
            mouse_high_question_pos = mouse_high_question.getPos()
            
            x_min = slider_high_question.pos[0] - slider_high_question.size[0] / 2
            x_max = slider_high_question.pos[0] + slider_high_question.size[0] / 2
            y_min = slider_high_question.pos[1] - 0.1
            y_max = slider_high_question.pos[1] + 0.1
            
            # Move marker when hovering over the slider
            if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                slider_high_question.markerPos = slider_high_question._posToRating(mouse_high_question_pos)
            
            buttonState = mouse_high_question.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_high_question_pos[0] <= x_max and y_min <= mouse_high_question_pos[1] <= y_max:
                    slider_high_question.rating = slider_high_question.markerPos
                    continueRoutine = False
            
            prevButtonState = buttonState
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=question_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                question_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if question_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in question_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "question_high" ---
        for thisComponent in question_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for question_high
        question_high.tStop = globalClock.getTime(format='float')
        question_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('question_high.stopped', question_high.tStop)
        # store data for high_3 (TrialHandler)
        # Run 'End Routine' code from code_high_question
        thisExp.addData('trial', block_trial)
        thisExp.addData('answ_high_question', slider_high_question.markerPos)
        
        if slider_high_question.markerPos is not None:
            trust_answer = slider_high_question.markerPos
        else:
            trust_answer = "NA"
        
        marker = f"CONF_H | B{block} | T{block_trial} | V={trust_answer}"
        outlet.push_sample([marker])
        # the Routine "question_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisHigh_3 as finished
        if hasattr(thisHigh_3, 'status'):
            thisHigh_3.status = FINISHED
        # if awaiting a pause, pause now
        if high_3.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            high_3.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'high_3'
    high_3.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "SATI_short_high_instructions" ---
    # create an object to store info about Routine SATI_short_high_instructions
    SATI_short_high_instructions = data.Routine(
        name='SATI_short_high_instructions',
        components=[text_SATI_short_high_instructions, key_resp_SATI_short_high_instructions, text_space_SATI_short_high_instructions],
    )
    SATI_short_high_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_SATI_short_high_instructions
    key_resp_SATI_short_high_instructions.keys = []
    key_resp_SATI_short_high_instructions.rt = []
    _key_resp_SATI_short_high_instructions_allKeys = []
    # store start times for SATI_short_high_instructions
    SATI_short_high_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    SATI_short_high_instructions.tStart = globalClock.getTime(format='float')
    SATI_short_high_instructions.status = STARTED
    thisExp.addData('SATI_short_high_instructions.started', SATI_short_high_instructions.tStart)
    SATI_short_high_instructions.maxDuration = None
    # keep track of which components have finished
    SATI_short_high_instructionsComponents = SATI_short_high_instructions.components
    for thisComponent in SATI_short_high_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "SATI_short_high_instructions" ---
    thisExp.currentRoutine = SATI_short_high_instructions
    SATI_short_high_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_SATI_short_high_instructions* updates
        
        # if text_SATI_short_high_instructions is starting this frame...
        if text_SATI_short_high_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_SATI_short_high_instructions.started')
            # update status
            text_SATI_short_high_instructions.status = STARTED
            text_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_SATI_short_high_instructions is active this frame...
        if text_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_SATI_short_high_instructions* updates
        
        # if key_resp_SATI_short_high_instructions is starting this frame...
        if key_resp_SATI_short_high_instructions.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            key_resp_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_SATI_short_high_instructions.status = STARTED
            # keyboard checking is just starting
            key_resp_SATI_short_high_instructions.clock.reset()  # now t=0
        if key_resp_SATI_short_high_instructions.status == STARTED:
            theseKeys = key_resp_SATI_short_high_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_SATI_short_high_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_SATI_short_high_instructions_allKeys):
                key_resp_SATI_short_high_instructions.keys = _key_resp_SATI_short_high_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_SATI_short_high_instructions.rt = _key_resp_SATI_short_high_instructions_allKeys[-1].rt
                key_resp_SATI_short_high_instructions.duration = _key_resp_SATI_short_high_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_SATI_short_high_instructions* updates
        
        # if text_space_SATI_short_high_instructions is starting this frame...
        if text_space_SATI_short_high_instructions.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_SATI_short_high_instructions.frameNStart = frameN  # exact frame index
            text_space_SATI_short_high_instructions.tStart = t  # local t and not account for scr refresh
            text_space_SATI_short_high_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_SATI_short_high_instructions, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_SATI_short_high_instructions.status = STARTED
            text_space_SATI_short_high_instructions.setAutoDraw(True)
        
        # if text_space_SATI_short_high_instructions is active this frame...
        if text_space_SATI_short_high_instructions.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=SATI_short_high_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            SATI_short_high_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if SATI_short_high_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in SATI_short_high_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "SATI_short_high_instructions" ---
    for thisComponent in SATI_short_high_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for SATI_short_high_instructions
    SATI_short_high_instructions.tStop = globalClock.getTime(format='float')
    SATI_short_high_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('SATI_short_high_instructions.stopped', SATI_short_high_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "SATI_short_high_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    SATI_alta_3 = data.TrialHandler2(
        name='SATI_alta_3',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('SATI.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(SATI_alta_3)  # add the loop to the experiment
    thisSATI_alta_3 = SATI_alta_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSATI_alta_3.rgb)
    if thisSATI_alta_3 != None:
        for paramName in thisSATI_alta_3:
            globals()[paramName] = thisSATI_alta_3[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSATI_alta_3 in SATI_alta_3:
        SATI_alta_3.status = STARTED
        if hasattr(thisSATI_alta_3, 'status'):
            thisSATI_alta_3.status = STARTED
        currentLoop = SATI_alta_3
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSATI_alta_3.rgb)
        if thisSATI_alta_3 != None:
            for paramName in thisSATI_alta_3:
                globals()[paramName] = thisSATI_alta_3[paramName]
        
        # --- Prepare to start Routine "SATI_short_high" ---
        # create an object to store info about Routine SATI_short_high
        SATI_short_high = data.Routine(
            name='SATI_short_high',
            components=[text_SATI_short_high, text_SATI_short_high_clarification, slider_SATI_short_high, mouse_SATI_short_high, prog_short_high],
        )
        SATI_short_high.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_SATI_short_high.setText(Question)
        text_SATI_short_high_clarification.setText(Clarification)
        slider_SATI_short_high.reset()
        # setup some python lists for storing info about the mouse_SATI_short_high
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_SATI_short_high
        # Reset mouse clicks
        mouse_SATI_short_high.clickReset()
        
        # Previous button state (important for detecting a clean click)
        prevButtonState = mouse_SATI_short_high.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_short_high.setProgress((block_trial - 1) / 6)
        # store start times for SATI_short_high
        SATI_short_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        SATI_short_high.tStart = globalClock.getTime(format='float')
        SATI_short_high.status = STARTED
        thisExp.addData('SATI_short_high.started', SATI_short_high.tStart)
        SATI_short_high.maxDuration = None
        # keep track of which components have finished
        SATI_short_highComponents = SATI_short_high.components
        for thisComponent in SATI_short_high.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "SATI_short_high" ---
        thisExp.currentRoutine = SATI_short_high
        SATI_short_high.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisSATI_alta_3, 'status') and thisSATI_alta_3.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_SATI_short_high* updates
            
            # if text_SATI_short_high is starting this frame...
            if text_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high.frameNStart = frameN  # exact frame index
                text_SATI_short_high.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high.status = STARTED
                text_SATI_short_high.setAutoDraw(True)
            
            # if text_SATI_short_high is active this frame...
            if text_SATI_short_high.status == STARTED:
                # update params
                pass
            
            # *text_SATI_short_high_clarification* updates
            
            # if text_SATI_short_high_clarification is starting this frame...
            if text_SATI_short_high_clarification.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_SATI_short_high_clarification.frameNStart = frameN  # exact frame index
                text_SATI_short_high_clarification.tStart = t  # local t and not account for scr refresh
                text_SATI_short_high_clarification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_SATI_short_high_clarification, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_SATI_short_high_clarification.status = STARTED
                text_SATI_short_high_clarification.setAutoDraw(True)
            
            # if text_SATI_short_high_clarification is active this frame...
            if text_SATI_short_high_clarification.status == STARTED:
                # update params
                pass
            
            # *slider_SATI_short_high* updates
            
            # if slider_SATI_short_high is starting this frame...
            if slider_SATI_short_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_SATI_short_high.frameNStart = frameN  # exact frame index
                slider_SATI_short_high.tStart = t  # local t and not account for scr refresh
                slider_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_SATI_short_high.started')
                # update status
                slider_SATI_short_high.status = STARTED
                slider_SATI_short_high.setAutoDraw(True)
            
            # if slider_SATI_short_high is active this frame...
            if slider_SATI_short_high.status == STARTED:
                # update params
                pass
            # *mouse_SATI_short_high* updates
            
            # if mouse_SATI_short_high is starting this frame...
            if mouse_SATI_short_high.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_SATI_short_high.frameNStart = frameN  # exact frame index
                mouse_SATI_short_high.tStart = t  # local t and not account for scr refresh
                mouse_SATI_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_SATI_short_high, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_SATI_short_high.status = STARTED
                mouse_SATI_short_high.mouseClock.reset()
                prevButtonState = mouse_SATI_short_high.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_SATI_short_high
            mouse_pos = mouse_SATI_short_high.getPos()
            
            x_min = slider_SATI_short_high.pos[0] - slider_SATI_short_high.size[0] / 2
            x_max = slider_SATI_short_high.pos[0] + slider_SATI_short_high.size[0] / 2
            y_min = slider_SATI_short_high.pos[1] - 0.1
            y_max = slider_SATI_short_high.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_SATI_short_high.markerPos = slider_SATI_short_high._posToRating(mouse_pos)
            
            buttonState = mouse_SATI_short_high.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_SATI_short_high.rating = slider_SATI_short_high.markerPos
                    responseSelected = True
            
            # Wait until the mouse has been released for 0.8 s
            if responseSelected:
                if buttonState == 0:
                    if releaseStartTime is None:
                        releaseStartTime = t
                    elif t - releaseStartTime >= releaseWaitTime:
                        continueRoutine = False
                else:
                    releaseStartTime = None
            
            prevButtonState = buttonState
            
            # *prog_short_high* updates
            
            # if prog_short_high is starting this frame...
            if prog_short_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_short_high.frameNStart = frameN  # exact frame index
                prog_short_high.tStart = t  # local t and not account for scr refresh
                prog_short_high.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_short_high, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_short_high.started')
                # update status
                prog_short_high.status = STARTED
                prog_short_high.setAutoDraw(True)
            
            # if prog_short_high is active this frame...
            if prog_short_high.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=SATI_short_high,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                SATI_short_high.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if SATI_short_high.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in SATI_short_high.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "SATI_short_high" ---
        for thisComponent in SATI_short_high.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for SATI_short_high
        SATI_short_high.tStop = globalClock.getTime(format='float')
        SATI_short_high.tStopRefresh = tThisFlipGlobal
        thisExp.addData('SATI_short_high.stopped', SATI_short_high.tStop)
        # store data for SATI_alta_3 (TrialHandler)
        # Run 'End Routine' code from code_SATI_short_high
        if slider_SATI_short_high.markerPos is not None:
            thisExp.addData('answ_SATI_short_high', slider_SATI_short_high.markerPos)
        else:
            thisExp.addData('answ_SATI_short_high', 'NA')
        # the Routine "SATI_short_high" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisSATI_alta_3 as finished
        if hasattr(thisSATI_alta_3, 'status'):
            thisSATI_alta_3.status = FINISHED
        # if awaiting a pause, pause now
        if SATI_alta_3.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            SATI_alta_3.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'SATI_alta_3'
    SATI_alta_3.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "how_you_feel_high" ---
    # create an object to store info about Routine how_you_feel_high
    how_you_feel_high = data.Routine(
        name='how_you_feel_high',
        components=[pleasant_high, unpleasant_high, text_how_you_feel_high],
    )
    how_you_feel_high.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset pleasant_high to account for continued clicks & clear times on/off
    pleasant_high.reset()
    # reset unpleasant_high to account for continued clicks & clear times on/off
    unpleasant_high.reset()
    # store start times for how_you_feel_high
    how_you_feel_high.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    how_you_feel_high.tStart = globalClock.getTime(format='float')
    how_you_feel_high.status = STARTED
    thisExp.addData('how_you_feel_high.started', how_you_feel_high.tStart)
    how_you_feel_high.maxDuration = None
    # keep track of which components have finished
    how_you_feel_highComponents = how_you_feel_high.components
    for thisComponent in how_you_feel_high.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "how_you_feel_high" ---
    thisExp.currentRoutine = how_you_feel_high
    how_you_feel_high.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pleasant_high* updates
        
        # if pleasant_high is starting this frame...
        if pleasant_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            pleasant_high.frameNStart = frameN  # exact frame index
            pleasant_high.tStart = t  # local t and not account for scr refresh
            pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            pleasant_high.status = STARTED
            win.callOnFlip(pleasant_high.buttonClock.reset)
            pleasant_high.setAutoDraw(True)
        
        # if pleasant_high is active this frame...
        if pleasant_high.status == STARTED:
            # update params
            pass
            # check whether pleasant_high has been pressed
            if pleasant_high.isClicked:
                if not pleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    pleasant_high.timesOn.append(pleasant_high.buttonClock.getTime())
                    pleasant_high.timesOff.append(pleasant_high.buttonClock.getTime())
                elif len(pleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    pleasant_high.timesOff[-1] = pleasant_high.buttonClock.getTime()
                if not pleasant_high.wasClicked:
                    # end routine when pleasant_high is clicked
                    continueRoutine = False
                if not pleasant_high.wasClicked:
                    # run callback code when pleasant_high is clicked
                    pass
        # take note of whether pleasant_high was clicked, so that next frame we know if clicks are new
        pleasant_high.wasClicked = pleasant_high.isClicked and pleasant_high.status == STARTED
        # *unpleasant_high* updates
        
        # if unpleasant_high is starting this frame...
        if unpleasant_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            unpleasant_high.frameNStart = frameN  # exact frame index
            unpleasant_high.tStart = t  # local t and not account for scr refresh
            unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'unpleasant_high.started')
            # update status
            unpleasant_high.status = STARTED
            win.callOnFlip(unpleasant_high.buttonClock.reset)
            unpleasant_high.setAutoDraw(True)
        
        # if unpleasant_high is active this frame...
        if unpleasant_high.status == STARTED:
            # update params
            pass
            # check whether unpleasant_high has been pressed
            if unpleasant_high.isClicked:
                if not unpleasant_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    unpleasant_high.timesOn.append(unpleasant_high.buttonClock.getTime())
                    unpleasant_high.timesOff.append(unpleasant_high.buttonClock.getTime())
                elif len(unpleasant_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    unpleasant_high.timesOff[-1] = unpleasant_high.buttonClock.getTime()
                if not unpleasant_high.wasClicked:
                    # end routine when unpleasant_high is clicked
                    continueRoutine = False
                if not unpleasant_high.wasClicked:
                    # run callback code when unpleasant_high is clicked
                    pass
        # take note of whether unpleasant_high was clicked, so that next frame we know if clicks are new
        unpleasant_high.wasClicked = unpleasant_high.isClicked and unpleasant_high.status == STARTED
        
        # *text_how_you_feel_high* updates
        
        # if text_how_you_feel_high is starting this frame...
        if text_how_you_feel_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_how_you_feel_high.frameNStart = frameN  # exact frame index
            text_how_you_feel_high.tStart = t  # local t and not account for scr refresh
            text_how_you_feel_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_how_you_feel_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_how_you_feel_high.started')
            # update status
            text_how_you_feel_high.status = STARTED
            text_how_you_feel_high.setAutoDraw(True)
        
        # if text_how_you_feel_high is active this frame...
        if text_how_you_feel_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=how_you_feel_high,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            how_you_feel_high.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if how_you_feel_high.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in how_you_feel_high.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "how_you_feel_high" ---
    for thisComponent in how_you_feel_high.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for how_you_feel_high
    how_you_feel_high.tStop = globalClock.getTime(format='float')
    how_you_feel_high.tStopRefresh = tThisFlipGlobal
    thisExp.addData('how_you_feel_high.stopped', how_you_feel_high.tStop)
    thisExp.addData('pleasant_high.numClicks', pleasant_high.numClicks)
    if pleasant_high.numClicks:
       thisExp.addData('pleasant_high.timesOn', pleasant_high.timesOn)
       thisExp.addData('pleasant_high.timesOff', pleasant_high.timesOff)
    else:
       thisExp.addData('pleasant_high.timesOn', "")
       thisExp.addData('pleasant_high.timesOff', "")
    thisExp.addData('unpleasant_high.numClicks', unpleasant_high.numClicks)
    if unpleasant_high.numClicks:
       thisExp.addData('unpleasant_high.timesOn', unpleasant_high.timesOn)
       thisExp.addData('unpleasant_high.timesOff', unpleasant_high.timesOff)
    else:
       thisExp.addData('unpleasant_high.timesOn', "")
       thisExp.addData('unpleasant_high.timesOff', "")
    # Run 'End Routine' code from code_how_you_feel_high
    if pleasant_high.isClicked:
        valence_answer = "pleasant"
    elif unpleasant_high.isClicked:
        valence_answer = "unpleasant"
    else:
        valence_answer = "no_response"
    
    thisExp.addData('answ_valence', valence_answer)
    thisExp.nextEntry()
    # the Routine "how_you_feel_high" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "pleasant_high_choice" ---
    # create an object to store info about Routine pleasant_high_choice
    pleasant_high_choice = data.Routine(
        name='pleasant_high_choice',
        components=[excited_high, relaxed_high, text_pleasant_high],
    )
    pleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset excited_high to account for continued clicks & clear times on/off
    excited_high.reset()
    # reset relaxed_high to account for continued clicks & clear times on/off
    relaxed_high.reset()
    # Run 'Begin Routine' code from code_pleasant_high
    # Only show this routine if "pleasant" was selected
    if valence_answer != "pleasant":
        continueRoutine = False
    # store start times for pleasant_high_choice
    pleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    pleasant_high_choice.tStart = globalClock.getTime(format='float')
    pleasant_high_choice.status = STARTED
    thisExp.addData('pleasant_high_choice.started', pleasant_high_choice.tStart)
    pleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    pleasant_high_choiceComponents = pleasant_high_choice.components
    for thisComponent in pleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "pleasant_high_choice" ---
    thisExp.currentRoutine = pleasant_high_choice
    pleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *excited_high* updates
        
        # if excited_high is starting this frame...
        if excited_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            excited_high.frameNStart = frameN  # exact frame index
            excited_high.tStart = t  # local t and not account for scr refresh
            excited_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(excited_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            excited_high.status = STARTED
            win.callOnFlip(excited_high.buttonClock.reset)
            excited_high.setAutoDraw(True)
        
        # if excited_high is active this frame...
        if excited_high.status == STARTED:
            # update params
            pass
            # check whether excited_high has been pressed
            if excited_high.isClicked:
                if not excited_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    excited_high.timesOn.append(excited_high.buttonClock.getTime())
                    excited_high.timesOff.append(excited_high.buttonClock.getTime())
                elif len(excited_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    excited_high.timesOff[-1] = excited_high.buttonClock.getTime()
                if not excited_high.wasClicked:
                    # end routine when excited_high is clicked
                    continueRoutine = False
                if not excited_high.wasClicked:
                    # run callback code when excited_high is clicked
                    pass
        # take note of whether excited_high was clicked, so that next frame we know if clicks are new
        excited_high.wasClicked = excited_high.isClicked and excited_high.status == STARTED
        # *relaxed_high* updates
        
        # if relaxed_high is starting this frame...
        if relaxed_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            relaxed_high.frameNStart = frameN  # exact frame index
            relaxed_high.tStart = t  # local t and not account for scr refresh
            relaxed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(relaxed_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'relaxed_high.started')
            # update status
            relaxed_high.status = STARTED
            win.callOnFlip(relaxed_high.buttonClock.reset)
            relaxed_high.setAutoDraw(True)
        
        # if relaxed_high is active this frame...
        if relaxed_high.status == STARTED:
            # update params
            pass
            # check whether relaxed_high has been pressed
            if relaxed_high.isClicked:
                if not relaxed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    relaxed_high.timesOn.append(relaxed_high.buttonClock.getTime())
                    relaxed_high.timesOff.append(relaxed_high.buttonClock.getTime())
                elif len(relaxed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    relaxed_high.timesOff[-1] = relaxed_high.buttonClock.getTime()
                if not relaxed_high.wasClicked:
                    # end routine when relaxed_high is clicked
                    continueRoutine = False
                if not relaxed_high.wasClicked:
                    # run callback code when relaxed_high is clicked
                    pass
        # take note of whether relaxed_high was clicked, so that next frame we know if clicks are new
        relaxed_high.wasClicked = relaxed_high.isClicked and relaxed_high.status == STARTED
        
        # *text_pleasant_high* updates
        
        # if text_pleasant_high is starting this frame...
        if text_pleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pleasant_high.frameNStart = frameN  # exact frame index
            text_pleasant_high.tStart = t  # local t and not account for scr refresh
            text_pleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pleasant_high.started')
            # update status
            text_pleasant_high.status = STARTED
            text_pleasant_high.setAutoDraw(True)
        
        # if text_pleasant_high is active this frame...
        if text_pleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=pleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            pleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if pleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in pleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "pleasant_high_choice" ---
    for thisComponent in pleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for pleasant_high_choice
    pleasant_high_choice.tStop = globalClock.getTime(format='float')
    pleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('pleasant_high_choice.stopped', pleasant_high_choice.tStop)
    thisExp.addData('excited_high.numClicks', excited_high.numClicks)
    if excited_high.numClicks:
       thisExp.addData('excited_high.timesOn', excited_high.timesOn)
       thisExp.addData('excited_high.timesOff', excited_high.timesOff)
    else:
       thisExp.addData('excited_high.timesOn', "")
       thisExp.addData('excited_high.timesOff', "")
    thisExp.addData('relaxed_high.numClicks', relaxed_high.numClicks)
    if relaxed_high.numClicks:
       thisExp.addData('relaxed_high.timesOn', relaxed_high.timesOn)
       thisExp.addData('relaxed_high.timesOff', relaxed_high.timesOff)
    else:
       thisExp.addData('relaxed_high.timesOn', "")
       thisExp.addData('relaxed_high.timesOff', "")
    # Run 'End Routine' code from code_pleasant_high
    if excited_high.isClicked:
        pleasant_answer = "excited"
    elif relaxed_high.isClicked:
        pleasant_answer = "relaxed"
    else:
        pleasant_answer = "no_response"
    
    thisExp.addData('answ_pleasant', pleasant_answer)
    thisExp.nextEntry()
    # the Routine "pleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "unpleasant_high_choice" ---
    # create an object to store info about Routine unpleasant_high_choice
    unpleasant_high_choice = data.Routine(
        name='unpleasant_high_choice',
        components=[stressed_high, bored_high, text_unpleasant_high],
    )
    unpleasant_high_choice.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # reset stressed_high to account for continued clicks & clear times on/off
    stressed_high.reset()
    # reset bored_high to account for continued clicks & clear times on/off
    bored_high.reset()
    # Run 'Begin Routine' code from code_unpleasant_high
    # Only show this routine if "unpleasant" was selected
    if valence_answer != "unpleasant":
        continueRoutine = False
    # store start times for unpleasant_high_choice
    unpleasant_high_choice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    unpleasant_high_choice.tStart = globalClock.getTime(format='float')
    unpleasant_high_choice.status = STARTED
    thisExp.addData('unpleasant_high_choice.started', unpleasant_high_choice.tStart)
    unpleasant_high_choice.maxDuration = None
    # keep track of which components have finished
    unpleasant_high_choiceComponents = unpleasant_high_choice.components
    for thisComponent in unpleasant_high_choice.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "unpleasant_high_choice" ---
    thisExp.currentRoutine = unpleasant_high_choice
    unpleasant_high_choice.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *stressed_high* updates
        
        # if stressed_high is starting this frame...
        if stressed_high.status == NOT_STARTED and t >= 0-frameTolerance:
            # keep track of start time/frame for later
            stressed_high.frameNStart = frameN  # exact frame index
            stressed_high.tStart = t  # local t and not account for scr refresh
            stressed_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(stressed_high, 'tStartRefresh')  # time at next scr refresh
            # update status
            stressed_high.status = STARTED
            win.callOnFlip(stressed_high.buttonClock.reset)
            stressed_high.setAutoDraw(True)
        
        # if stressed_high is active this frame...
        if stressed_high.status == STARTED:
            # update params
            pass
            # check whether stressed_high has been pressed
            if stressed_high.isClicked:
                if not stressed_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    stressed_high.timesOn.append(stressed_high.buttonClock.getTime())
                    stressed_high.timesOff.append(stressed_high.buttonClock.getTime())
                elif len(stressed_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    stressed_high.timesOff[-1] = stressed_high.buttonClock.getTime()
                if not stressed_high.wasClicked:
                    # end routine when stressed_high is clicked
                    continueRoutine = False
                if not stressed_high.wasClicked:
                    # run callback code when stressed_high is clicked
                    pass
        # take note of whether stressed_high was clicked, so that next frame we know if clicks are new
        stressed_high.wasClicked = stressed_high.isClicked and stressed_high.status == STARTED
        # *bored_high* updates
        
        # if bored_high is starting this frame...
        if bored_high.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            bored_high.frameNStart = frameN  # exact frame index
            bored_high.tStart = t  # local t and not account for scr refresh
            bored_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bored_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bored_high.started')
            # update status
            bored_high.status = STARTED
            win.callOnFlip(bored_high.buttonClock.reset)
            bored_high.setAutoDraw(True)
        
        # if bored_high is active this frame...
        if bored_high.status == STARTED:
            # update params
            pass
            # check whether bored_high has been pressed
            if bored_high.isClicked:
                if not bored_high.wasClicked:
                    # if this is a new click, store time of first click and clicked until
                    bored_high.timesOn.append(bored_high.buttonClock.getTime())
                    bored_high.timesOff.append(bored_high.buttonClock.getTime())
                elif len(bored_high.timesOff):
                    # if click is continuing from last frame, update time of clicked until
                    bored_high.timesOff[-1] = bored_high.buttonClock.getTime()
                if not bored_high.wasClicked:
                    # end routine when bored_high is clicked
                    continueRoutine = False
                if not bored_high.wasClicked:
                    # run callback code when bored_high is clicked
                    pass
        # take note of whether bored_high was clicked, so that next frame we know if clicks are new
        bored_high.wasClicked = bored_high.isClicked and bored_high.status == STARTED
        
        # *text_unpleasant_high* updates
        
        # if text_unpleasant_high is starting this frame...
        if text_unpleasant_high.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_unpleasant_high.frameNStart = frameN  # exact frame index
            text_unpleasant_high.tStart = t  # local t and not account for scr refresh
            text_unpleasant_high.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_unpleasant_high, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_unpleasant_high.started')
            # update status
            text_unpleasant_high.status = STARTED
            text_unpleasant_high.setAutoDraw(True)
        
        # if text_unpleasant_high is active this frame...
        if text_unpleasant_high.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=unpleasant_high_choice,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            unpleasant_high_choice.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if unpleasant_high_choice.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in unpleasant_high_choice.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "unpleasant_high_choice" ---
    for thisComponent in unpleasant_high_choice.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for unpleasant_high_choice
    unpleasant_high_choice.tStop = globalClock.getTime(format='float')
    unpleasant_high_choice.tStopRefresh = tThisFlipGlobal
    thisExp.addData('unpleasant_high_choice.stopped', unpleasant_high_choice.tStop)
    thisExp.addData('stressed_high.numClicks', stressed_high.numClicks)
    if stressed_high.numClicks:
       thisExp.addData('stressed_high.timesOn', stressed_high.timesOn)
       thisExp.addData('stressed_high.timesOff', stressed_high.timesOff)
    else:
       thisExp.addData('stressed_high.timesOn', "")
       thisExp.addData('stressed_high.timesOff', "")
    thisExp.addData('bored_high.numClicks', bored_high.numClicks)
    if bored_high.numClicks:
       thisExp.addData('bored_high.timesOn', bored_high.timesOn)
       thisExp.addData('bored_high.timesOff', bored_high.timesOff)
    else:
       thisExp.addData('bored_high.timesOn', "")
       thisExp.addData('bored_high.timesOff', "")
    # Run 'End Routine' code from code_unpleasant_high
    if stressed_high.isClicked:
        unpleasant_answer = "stressed"
    elif bored_high.isClicked:
        unpleasant_answer = "bored"
    else:
        unpleasant_answer = "no_response"
    
    thisExp.addData('answ_unpleasant', unpleasant_answer)
    thisExp.nextEntry()
    # the Routine "unpleasant_high_choice" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "test_final_instructions" ---
    # create an object to store info about Routine test_final_instructions
    test_final_instructions = data.Routine(
        name='test_final_instructions',
        components=[text_test_final_instructions, key_resp_test_final_instrucciones, text_space_test_final_instrucciones],
    )
    test_final_instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_test_final_instrucciones
    key_resp_test_final_instrucciones.keys = []
    key_resp_test_final_instrucciones.rt = []
    _key_resp_test_final_instrucciones_allKeys = []
    # store start times for test_final_instructions
    test_final_instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    test_final_instructions.tStart = globalClock.getTime(format='float')
    test_final_instructions.status = STARTED
    thisExp.addData('test_final_instructions.started', test_final_instructions.tStart)
    test_final_instructions.maxDuration = None
    # keep track of which components have finished
    test_final_instructionsComponents = test_final_instructions.components
    for thisComponent in test_final_instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "test_final_instructions" ---
    thisExp.currentRoutine = test_final_instructions
    test_final_instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_test_final_instructions* updates
        
        # if text_test_final_instructions is starting this frame...
        if text_test_final_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_test_final_instructions.frameNStart = frameN  # exact frame index
            text_test_final_instructions.tStart = t  # local t and not account for scr refresh
            text_test_final_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_test_final_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_test_final_instructions.started')
            # update status
            text_test_final_instructions.status = STARTED
            text_test_final_instructions.setAutoDraw(True)
        
        # if text_test_final_instructions is active this frame...
        if text_test_final_instructions.status == STARTED:
            # update params
            pass
        
        # *key_resp_test_final_instrucciones* updates
        
        # if key_resp_test_final_instrucciones is starting this frame...
        if key_resp_test_final_instrucciones.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_test_final_instrucciones.frameNStart = frameN  # exact frame index
            key_resp_test_final_instrucciones.tStart = t  # local t and not account for scr refresh
            key_resp_test_final_instrucciones.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_test_final_instrucciones, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_test_final_instrucciones.status = STARTED
            # keyboard checking is just starting
            key_resp_test_final_instrucciones.clock.reset()  # now t=0
        if key_resp_test_final_instrucciones.status == STARTED:
            theseKeys = key_resp_test_final_instrucciones.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_test_final_instrucciones_allKeys.extend(theseKeys)
            if len(_key_resp_test_final_instrucciones_allKeys):
                key_resp_test_final_instrucciones.keys = _key_resp_test_final_instrucciones_allKeys[-1].name  # just the last key pressed
                key_resp_test_final_instrucciones.rt = _key_resp_test_final_instrucciones_allKeys[-1].rt
                key_resp_test_final_instrucciones.duration = _key_resp_test_final_instrucciones_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_test_final_instrucciones* updates
        
        # if text_space_test_final_instrucciones is starting this frame...
        if text_space_test_final_instrucciones.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_test_final_instrucciones.frameNStart = frameN  # exact frame index
            text_space_test_final_instrucciones.tStart = t  # local t and not account for scr refresh
            text_space_test_final_instrucciones.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_test_final_instrucciones, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_test_final_instrucciones.status = STARTED
            text_space_test_final_instrucciones.setAutoDraw(True)
        
        # if text_space_test_final_instrucciones is active this frame...
        if text_space_test_final_instrucciones.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=test_final_instructions,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            test_final_instructions.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if test_final_instructions.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in test_final_instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "test_final_instructions" ---
    for thisComponent in test_final_instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for test_final_instructions
    test_final_instructions.tStop = globalClock.getTime(format='float')
    test_final_instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('test_final_instructions.stopped', test_final_instructions.tStop)
    thisExp.nextEntry()
    # the Routine "test_final_instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    final = data.TrialHandler2(
        name='final',
        nReps=1, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('peidong_mei.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(final)  # add the loop to the experiment
    thisFinal = final.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisFinal.rgb)
    if thisFinal != None:
        for paramName in thisFinal:
            globals()[paramName] = thisFinal[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisFinal in final:
        final.status = STARTED
        if hasattr(thisFinal, 'status'):
            thisFinal.status = STARTED
        currentLoop = final
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisFinal.rgb)
        if thisFinal != None:
            for paramName in thisFinal:
                globals()[paramName] = thisFinal[paramName]
        
        # --- Prepare to start Routine "test_final" ---
        # create an object to store info about Routine test_final
        test_final = data.Routine(
            name='test_final',
            components=[text_test_final, text_test_final_clarification, slider_test_final, mouse_test_final, prog_final],
        )
        test_final.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_test_final.setText(Question)
        text_test_final_clarification.setText(Clarification)
        slider_test_final.reset()
        # setup some python lists for storing info about the mouse_test_final
        gotValidClick = False  # until a click is received
        # Run 'Begin Routine' code from code_test_final
        # Reset mouse clicks
        mouse_test_final.clickReset()
        
        # Previous button state
        prevButtonState = mouse_test_final.getPressed()[0]
        
        # Control variable to avoid inheriting previous clicks
        readyForClick = False
        
        responseSelected = False
        releaseStartTime = None
        releaseWaitTime = 0.3
        prog_final.setProgress(final.thisN / 8)
        # store start times for test_final
        test_final.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test_final.tStart = globalClock.getTime(format='float')
        test_final.status = STARTED
        thisExp.addData('test_final.started', test_final.tStart)
        test_final.maxDuration = None
        # keep track of which components have finished
        test_finalComponents = test_final.components
        for thisComponent in test_final.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test_final" ---
        thisExp.currentRoutine = test_final
        test_final.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisFinal, 'status') and thisFinal.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_test_final* updates
            
            # if text_test_final is starting this frame...
            if text_test_final.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_test_final.frameNStart = frameN  # exact frame index
                text_test_final.tStart = t  # local t and not account for scr refresh
                text_test_final.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_test_final, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_test_final.status = STARTED
                text_test_final.setAutoDraw(True)
            
            # if text_test_final is active this frame...
            if text_test_final.status == STARTED:
                # update params
                pass
            
            # *text_test_final_clarification* updates
            
            # if text_test_final_clarification is starting this frame...
            if text_test_final_clarification.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_test_final_clarification.frameNStart = frameN  # exact frame index
                text_test_final_clarification.tStart = t  # local t and not account for scr refresh
                text_test_final_clarification.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_test_final_clarification, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_test_final_clarification.status = STARTED
                text_test_final_clarification.setAutoDraw(True)
            
            # if text_test_final_clarification is active this frame...
            if text_test_final_clarification.status == STARTED:
                # update params
                pass
            
            # *slider_test_final* updates
            
            # if slider_test_final is starting this frame...
            if slider_test_final.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_test_final.frameNStart = frameN  # exact frame index
                slider_test_final.tStart = t  # local t and not account for scr refresh
                slider_test_final.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_test_final, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_test_final.started')
                # update status
                slider_test_final.status = STARTED
                slider_test_final.setAutoDraw(True)
            
            # if slider_test_final is active this frame...
            if slider_test_final.status == STARTED:
                # update params
                pass
            # *mouse_test_final* updates
            
            # if mouse_test_final is starting this frame...
            if mouse_test_final.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_test_final.frameNStart = frameN  # exact frame index
                mouse_test_final.tStart = t  # local t and not account for scr refresh
                mouse_test_final.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_test_final, 'tStartRefresh')  # time at next scr refresh
                # update status
                mouse_test_final.status = STARTED
                mouse_test_final.mouseClock.reset()
                prevButtonState = mouse_test_final.getPressed()  # if button is down already this ISN'T a new click
            # Run 'Each Frame' code from code_test_final
            mouse_pos = mouse_test_final.getPos()
            
            x_min = slider_test_final.pos[0] - slider_test_final.size[0] / 2
            x_max = slider_test_final.pos[0] + slider_test_final.size[0] / 2
            y_min = slider_test_final.pos[1] - 0.1
            y_max = slider_test_final.pos[1] + 0.1
            
            # Move marker
            if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                slider_test_final.markerPos = slider_test_final._posToRating(mouse_pos)
            
            buttonState = mouse_test_final.getPressed()[0]
            
            # Wait until the mouse button is released
            if buttonState == 0:
                readyForClick = True
            
            # Accept only a clean click
            if readyForClick and buttonState == 1 and prevButtonState == 0:
                if x_min <= mouse_pos[0] <= x_max and y_min <= mouse_pos[1] <= y_max:
                    slider_test_final.rating = slider_test_final.markerPos
                    responseSelected = True
            
            # Wait until the mouse has been released for 0.8 s
            if responseSelected:
                if buttonState == 0:
                    if releaseStartTime is None:
                        releaseStartTime = t
                    elif t - releaseStartTime >= releaseWaitTime:
                        continueRoutine = False
                else:
                    releaseStartTime = None
            
            prevButtonState = buttonState
            
            # *prog_final* updates
            
            # if prog_final is starting this frame...
            if prog_final.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_final.frameNStart = frameN  # exact frame index
                prog_final.tStart = t  # local t and not account for scr refresh
                prog_final.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_final, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'prog_final.started')
                # update status
                prog_final.status = STARTED
                prog_final.setAutoDraw(True)
            
            # if prog_final is active this frame...
            if prog_final.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test_final,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                test_final.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if test_final.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in test_final.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test_final" ---
        for thisComponent in test_final.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test_final
        test_final.tStop = globalClock.getTime(format='float')
        test_final.tStopRefresh = tThisFlipGlobal
        thisExp.addData('test_final.stopped', test_final.tStop)
        # store data for final (TrialHandler)
        # Run 'End Routine' code from code_test_final
        thisExp.addData('trial', final.thisN + 1)
        
        # Save slider response
        thisExp.addData('answ_test_final', slider_test_final.markerPos)
        # the Routine "test_final" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisFinal as finished
        if hasattr(thisFinal, 'status'):
            thisFinal.status = FINISHED
        # if awaiting a pause, pause now
        if final.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            final.status = STARTED
        thisExp.nextEntry()
        
    # completed 1 repeats of 'final'
    final.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End" ---
    # create an object to store info about Routine End
    End = data.Routine(
        name='End',
        components=[text_end, key_resp_end, text_space_end],
    )
    End.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_end
    key_resp_end.keys = []
    key_resp_end.rt = []
    _key_resp_end_allKeys = []
    # store start times for End
    End.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End.tStart = globalClock.getTime(format='float')
    End.status = STARTED
    thisExp.addData('End.started', End.tStart)
    End.maxDuration = None
    # keep track of which components have finished
    EndComponents = End.components
    for thisComponent in End.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End" ---
    thisExp.currentRoutine = End
    End.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # *key_resp_end* updates
        
        # if key_resp_end is starting this frame...
        if key_resp_end.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_end.frameNStart = frameN  # exact frame index
            key_resp_end.tStart = t  # local t and not account for scr refresh
            key_resp_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_end, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_end.status = STARTED
            # keyboard checking is just starting
            key_resp_end.clock.reset()  # now t=0
        if key_resp_end.status == STARTED:
            theseKeys = key_resp_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_end_allKeys.extend(theseKeys)
            if len(_key_resp_end_allKeys):
                key_resp_end.keys = _key_resp_end_allKeys[-1].name  # just the last key pressed
                key_resp_end.rt = _key_resp_end_allKeys[-1].rt
                key_resp_end.duration = _key_resp_end_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_space_end* updates
        
        # if text_space_end is starting this frame...
        if text_space_end.status == NOT_STARTED and t >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            text_space_end.frameNStart = frameN  # exact frame index
            text_space_end.tStart = t  # local t and not account for scr refresh
            text_space_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_space_end, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_space_end.status = STARTED
            text_space_end.setAutoDraw(True)
        
        # if text_space_end is active this frame...
        if text_space_end.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=End,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            End.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if End.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in End.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End" ---
    for thisComponent in End.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End
    End.tStop = globalClock.getTime(format='float')
    End.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End.stopped', End.tStop)
    thisExp.nextEntry()
    # the Routine "End" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
