def reformat_command_output(command_output: dict) -> dict:
    reformatted_output = {
        "intent": command_output["intent"],
        "tags": {}
    }

    for tag, value in command_output["tags"]:
        if value not in reformatted_output["tags"]:
            reformatted_output["tags"][value] = []
        reformatted_output["tags"][value].append(tag)

    return reformatted_output

# Example usage:
command_output = {
    'intent': 'GetWeather',
    'tags': [('cedar', 'B-city'), ('rapids', 'I-city'), ('idaho', 'B-state'), ('tomorrow', 'I-state')]
}

formatted_output = reformat_command_output(command_output)
print(formatted_output)



from methods import DBUSMethods
import re
import json
from methods import GeneralMethods
from speak_responses import SpeakResponse
from thefuzz import process

generalMethods = GeneralMethods()


class CommandExecution:
    def __init__(self) -> None:
        self.dbusMethods = DBUSMethods()

    def messageSend(self, _name: str, _message: str, _send: bool = False):
        _message_send_payload = {
            "payload": "message/command",
            "data": {
                "command": "message",
                "contact": _name.strip(),
                "content": _message,
                "send": _send,
            },
        }

        _payload = json.dumps(_message_send_payload)

        print("This is message send payload: ", _payload, flush=True)
        try:
            _response = self.dbusMethods.executeDBus(_payload)
            print("This is response payload: ", _response, flush=True)
        except Exception as e:
            print(e, flush=True)
            print("Not able to send the message due to dbus problem", flush=True)

    def messageContactSearch(self, _contact_name: str) -> str:
        _contact_search_payload = {
            "payload": "message/command",
            "data": {"command": "message", "contact": _contact_name},
        }
        _payload = json.dumps(_contact_search_payload)
        _response = json.loads(self.dbusMethods.executeDBus(_payload))

        print("This is message payload", _payload)
        print("This is message related response: ", type(_response))
        print("This is message related response: ", _response)

        if _response["data"]["status"] != -1:
            print(
                "This is the contact list of search:",
                ",".join(_response["data"]["contacts"]),
            )

        if _response["data"]["status"] == -2:
            print("more than one contact found!!!")
            return f"""Contacts Found:
            {str(",".join(_response['data']['contacts']))}"""

        if _response["data"]["status"] == 1:
            print("Only one contact found!!!")
            return f"""Contact Found:
            {str(",".join(_response['data']['contacts']))}"""

        if _response["data"]["status"] == -1:
            return f"No Contacts Found!!!"

    def messageUpdate(self, _message: str):
        _message_update_payload = {
            "payload": "message/command",
            "data": {
                "command": "message",
                "content_append": _message,
            },
        }
        _payload = json.dumps(_message_update_payload)
        print("This is message update payload:", _payload)
        _response = self.dbusMethods.executeDBus(_payload)

        print("\nMessage Has been Updated!!!!!\n")



    def commands(self, command_list: list) -> None:
        speakResponse = SpeakResponse()
        dbusMethods = DBUSMethods(speakResponse)
        for _ in command_list:
            try:
                _ = _.strip()
                if _ == "":
                    continue

                elif _ == "start surgery":
                    # print("Command: start the surgery")
                    x = re.split("start surgery", _)[-1]
                    x2 = x[1:]
                    # NEW_TEXT += "Started the surgery session. "
                    dbusMethods.surgeryCommand("surgery_start", "NULL", "NULL")

                elif _ == "surgery complete":
                    # print("Command: surgery_completed")
                    x = re.split("surgery complete", _)[-1]
                    x2 = x[1:]
                    # NEW_TEXT += "Marked that, Surgery session has been completed. "
                    dbusMethods.surgeryCommand("surgery_complete", "NULL", "NULL")

                elif _ == "patient in":
                    # print("Command: patient in")
                    x = re.split("patient in", _)[-1]
                    x2 = x[1:]
                    # NEW_TEXT += "Okay, Noted that, patient is In. "
                    dbusMethods.surgeryCommand("patient_in", "NULL", "NULL")

                elif _ == "patient out":
                    # print("Command: patient out")
                    x = re.split("patient out", _)[-1]
                    x2 = x[1:]
                    # NEW_TEXT += "Marked down that patient is out. "
                    dbusMethods.surgeryCommand("patient_out", "NULL", "NULL")

                elif 0 == _.find("lower system volume"):
                    # print("Command: Lowering the volume")
                    x = re.split("lower system volume by", _)[-1]
                    x2 = x[1:]
                    temp = re.findall(r"\d+", x2)
                    string_value = "".join(temp)
                    volume = 0
                    try:
                        volume = int(string_value)
                    except ValueError as ve:
                        print("ValueError\n")
                    # NEW_TEXT += "Lowered down the system volume by "+ str(volume) + " ."
                    dbusMethods.systemVolumeCommand("Dec_sys_vol", volume)

                elif _.find("raise system volume") != -1:
                    # print("Command: Raise the volume")
                    x = re.split("raise system volume by", _)[-1]
                    x2 = x[1:]
                    temp = re.findall(r"\d+", x2)
                    string_value = "".join(temp)
                    volume = 0
                    try:
                        volume = int(string_value)
                    except ValueError as ve:
                        print("ValueError\n")
                    # NEW_TEXT += "Increased the system volume by "+ str(volume) + " ."
                    dbusMethods.systemVolumeCommand("Inc_sys_vol", volume)

                elif _.find("set system volume to") != -1:
                    x = re.split("set system volume to", _)[-1]
                    temp = x[1:]
                    string_value = "".join(temp)
                    volume = 0
                    try:
                        volume = int(string_value)
                    except ValueError as ve:
                        print("ValueError\n")
                    dbusMethods.systemVolumeCommand("Set_sys_vol", volume)

                elif 0 == _.find("show system volume"):
                    # print("Command: show system volume")
                    # NEW_TEXT += "Displayed system volume tray icon. "
                    dbusMethods.systemVolumeCommand("show_sys_vol", 0)

                elif 0 == _.find("hide system volume"):
                    # print("Command: show system volume")
                    # NEW_TEXT += "system volume tray icon is hidden now. "
                    dbusMethods.systemVolumeCommand("hide_sys_vol", 0)

                elif 0 == _.find("play song"):
                    # print("Command: Play song")
                    x = re.split("play song", _)[-1]
                    x2 = x[1:]
                    # NEW_TEXT += "Sure, I will play the "+x2+" song for you. "
                    dbusMethods.musicCommand("tracks", "NULL", x2)

                elif _ == "play music from recent library":
                    # print("Command: Playing songs from recent library")
                    # NEW_TEXT += "Playing the music from recent library. "
                    dbusMethods.musicCommand("recent", "NULL", "NULL")

                elif _ == "play music from favourite library":
                    # print("Command: Playing songs from favourite library")
                    # NEW_TEXT += "Playing the music from favorite library. "
                    dbusMethods.musicCommand("favorites", "NULL", "NULL")

                elif 0 == _.find("play music from"):
                    # print("Command: Playing songs from <SPECIFIED> playlist")
                    x = re.split("play music from", _)[-1]
                    x2 = x[1:]
                    yy = re.split("playlist", x2)
                    string_value = "".join(yy).strip()
                    search_result, accuracy = map(
                        list,
                        zip(
                            *process.extract(
                                string_value, dbusMethods.getPlayList(), limit=1
                            )
                        ),
                    )
                    print(search_result)
                    # NEW_TEXT += "Playing the music from" + str(string_value)+ "playlist. "
                    dbusMethods.musicCommand("playlists", "NULL", search_result[0])

                elif _ == "open spotify":
                    dbusMethods.musicCommand("open", "NULL", "NULL")

                elif _ == "close spotify":
                    dbusMethods.musicCommand("close", "NULL", "NULL")

                elif _ == "start music":
                    # print("Command: start music")
                    # NEW_TEXT += "Started the music for you. "
                    dbusMethods.musicCommand("Start", "NULL", "NULL")

                elif _ == "pause music":
                    # print("Command: stop music")
                    # NEW_TEXT += "Paused the music for you. "
                    dbusMethods.musicCommand("Pause", "NULL", "NULL")

                elif _ == "next song":
                    # print("Command: Play next song")
                    # NEW_TEXT += "Played the next song for you. "
                    dbusMethods.musicCommand("Next", "NULL", "NULL")

                elif _ == "previous song":
                    # print("Command: Play previous song")
                    # NEW_TEXT += "Started the Previous song for you. "
                    dbusMethods.musicCommand("Previous", "NULL", "NULL")

                elif _ == "increase music volume":
                    # print("Command: Increase music volume")
                    # NEW_TEXT += "Increased the music volume. "
                    dbusMethods.musicCommand("Inc_music", "NULL", "NULL")

                elif _ == "decrease music volume":
                    # print("Command: Decrease music volume")
                    # NEW_TEXT += "Decreased the music volume. "
                    dbusMethods.musicCommand("Dec_music", "NULL", "NULL")

                elif _ == "show music":
                    # print("Command: show music")
                    # NEW_TEXT += "Displayed the music tray icon for you. "
                    dbusMethods.musicCommand("Show_music", "NULL", "NULL")

                elif _ == "hide music":
                    # print("Command: hide music")
                    # NEW_TEXT += "The music tray icon is hidden now. "
                    dbusMethods.musicCommand("Hide_music", "NULL", "NULL")

                elif _ == "start stopwatch":
                    # print("Command: start the stopwatch")
                    # NEW_TEXT += " Started the stopwatch. "
                    dbusMethods.stopwatchCommand("Start_watch")

                elif _ in ["stop stopwatch", "pause stopwatch"]:
                    # print("Command: stop the stopwatch")
                    # NEW_TEXT += " Stopped the stopwatch. "
                    dbusMethods.stopwatchCommand("Stop_watch")

                elif _ == "reset stopwatch":
                    # print("Command: reset the stopwatch")
                    # NEW_TEXT += "Stopwatch is reset. "
                    dbusMethods.stopwatchCommand("Reset")

                elif _ == "show stopwatch":
                    # print("Command: show the stopwatch")
                    # NEW_TEXT += " Displayed the stopwatch tray icon. "
                    dbusMethods.stopwatchCommand("Show_watch")

                elif _ in ["hide stopwatch", "remove stopwatch"]:
                    # print("Command: hide the stopwatch")
                    # NEW_TEXT += "The stopwatch tray icon is hidden now. "
                    dbusMethods.stopwatchCommand("Hide_watch")

                elif _ == "restart stopwatch":
                    speakResponse.speakList.append(
                        "There is no such command to restart stopwatch"
                    )

                elif _ == "login":
                    # print("Command: Logging in")
                    dbusMethods.login()

                elif _ == "log me out":
                    # print("Command: Log me out")
                    # NEW_TEXT += "Logged you out. "
                    dbusMethods.logout()

                elif _ in [
                    "open dicom schedule",
                    "who is in my schedule",
                    "tell me who is in schedule",
                    "list all patient scheduled today",
                ]:
                    # print("Command: open dicom schedule")
                    # NEW_TEXT += "Opened the dicom schedule. "
                    dbusMethods.scheduleCommand("MWL_schedule_Today")

                elif _ == "show first patient scheduled":
                    # print("Command: show first patient scheduled")
                    # NEW_TEXT += "Here is the first patient scheduled. "
                    dbusMethods.scheduleCommand("MWL_First_Patient")

                elif _ == "show last patient scheduled":
                    # print("Command: show last patient scheduled")
                    # NEW_TEXT += "Here is the last patient scheduled. "
                    dbusMethods.scheduleCommand("MWL_Last_Patient")

                elif _ == "show next patient scheduled":
                    # print("Command: show next patient scheduled")
                    # NEW_TEXT += "Displayed the next patient scheduled. "
                    dbusMethods.scheduleCommand("MWL_Next_Patient")

                elif _ == "display temperature":
                    # print("Command: show temperature")
                    # NEW_TEXT += "Displayed the temperature. "
                    dbusMethods.temperatureCommand("temperature_show")

                elif _ == "hide temperature":
                    # print("Command: hide temperature")
                    # NEW_TEXT += "The temperature is hidden now. "
                    dbusMethods.temperatureCommand("temperature_hide")

                elif _ == "hide clock":
                    # print("Command: Hiding the clock")
                    # NEW_TEXT += "The clock is hidden now. "
                    # print(NEW_TEXT)
                    dbusMethods.clockCommand("Hide_clock")

                elif _ == "show clock" or _ == "display clock":
                    # print("Command: show the clock")
                    # NEW_TEXT += "The clock is now visible. "
                    dbusMethods.clockCommand("Display_clock")

                # elif _ == "start zoom call app":
                #     print("Command: start zoom call app")

                elif _.find("begin call with") != -1:
                    dbusMethods.zoomCommand(_.split("begin call with")[1].strip())

                elif _.find("in the zoom call") != -1:
                    dbusMethods.zoomCommand(
                        _.split("in the zoom call")[0].split("add")[1].strip()
                    )

                elif _ == "end call":
                    # print("Command: end call")
                    # NEW_TEXT += "Ended the call on your request. "
                    dbusMethods.executeDBusZoom("leave_call")

                elif _ == "mute mic":
                    # print("Command: mute mic")
                    # NEW_TEXT += "Muted the call on your request. "
                    dbusMethods.executeDBusZoom("Mute_conference")

                elif _ == "unmute mic":
                    # print("Command: unmute mic")
                    # NEW_TEXT += "Unmuted the call on your request. "
                    dbusMethods.executeDBusZoom("Unmute_conference")

                elif _ == "start video":
                    # print("Command: start video")
                    # NEW_TEXT += "Okay, Starting the video on your request. "
                    dbusMethods.executeDBusZoom("start_video")

                elif _ == "hide / stop video":
                    # print("Command: stop video")
                    # NEW_TEXT += "Okay, Stopping the video on your request. "
                    dbusMethods.executeDBusZoom("stop_video")

                elif _ in ["share surgery", "share surgical video"]:
                    dbusMethods.executeDBusZoom("share_SurgerySharing")

                elif _ in ["stop surgery", "stop surgical video"]:
                    dbusMethods.executeDBusZoom("stop_SurgerySharing")

                elif _ == "open dicom images":
                    # print("Command: open dicom image viewer app")
                    # NEW_TEXT += "Sure, Opened the DICOM Images App. "
                    dbusMethods.dicomCommand("openDicom")

                elif _ == "close dicom images":
                    # print("Command: close dicom image viewer app")
                    # NEW_TEXT += "The DICOM Image App is closed. "
                    dbusMethods.dicomCommand("closeDicom")

                elif _ == "open preference cards":
                    # NEW_TEXT += "Sure, Opened the pref card. "
                    dbusMethods.appOpenCloseCommands("prefcard", True, 3)

                elif _ in [
                    "enlarge the preference cards",
                    "open preference cards in enlarge mode",
                    "pop out the preference cards",
                ]:
                    # NEW_TEXT += "Sure, Enlarged the pref card. "
                    dbusMethods.appOpenCloseCommands("prefcard", True, 2)

                elif _ in [
                    "minimize the preference cards",
                    "pop in the preference cards",
                ]:
                    # NEW_TEXT += "Sure, minimized the pref card. "
                    dbusMethods.appOpenCloseCommands("prefcard", True, 1)

                elif _ == "close preference cards":
                    dbusMethods.appOpenCloseCommands("prefcard", False, 5)

                elif _ == "hide preference cards":
                    dbusMethods.appOpenCloseCommands("prefcard", False, 4)

                elif _ == "open ai dashboard":
                    dbusMethods.appOpenCloseCommands("aidashboard", True, 3)

                elif _ in [
                    "enlarge the ai dashboard",
                    "open ai dashboard in enlarge mode",
                    "pop out the ai dashboard",
                ]:
                    dbusMethods.appOpenCloseCommands("aidashboard", True, 2)

                elif _ in ["minimize the ai dashboard", "pop in the ai dashboard"]:
                    dbusMethods.appOpenCloseCommands("aidashboard", True, 1)

                elif _ == "close ai dashboard":
                    dbusMethods.appOpenCloseCommands("aidashboard", False, 5)

                elif _ == "hide ai dashboard":
                    dbusMethods.appOpenCloseCommands("aidashboard", False, 4)

                elif _ == "open up to date":
                    # NEW_TEXT += "Sure, Opened the uptodate. "
                    dbusMethods.appOpenCloseCommands("uptodate", True, 3)

                elif _ in [
                    "enlarge the up to date",
                    "open up to date in enlarge mode",
                    "pop out the up to date",
                ]:
                    # NEW_TEXT += "Sure, enlarged the uptodate. "
                    dbusMethods.appOpenCloseCommands("uptodate", True, 2)

                elif _ in ["minimize the up to date", "pop in the up to date"]:
                    # NEW_TEXT += "Sure, minimized the uptodate . "
                    dbusMethods.appOpenCloseCommands("uptodate", True, 1)

                elif _ == "close up to date":
                    dbusMethods.appOpenCloseCommands("uptodate", False, 5)

                elif _ == "hide up to date":
                    dbusMethods.appOpenCloseCommands("uptodate", False, 4)

                elif _ == "open nms routing":
                    dbusMethods.appOpenCloseCommands("nms", True, 3)

                elif _ in [
                    "enlarge the nms routing",
                    "open nms routing in enlarge mode",
                    "pop out the nms routing",
                ]:
                    dbusMethods.appOpenCloseCommands("nms", True, 2)

                elif _ in ["minimize the nms routing", "pop in the nms routing"]:
                    dbusMethods.appOpenCloseCommands("nms", True, 1)

                elif _ == "close nms routing":
                    dbusMethods.appOpenCloseCommands("nms", False, 5)

                elif _ == "hide nms routing":
                    dbusMethods.appOpenCloseCommands("nms", False, 4)

                elif _ == "open browser":
                    # NEW_TEXT += "Sure, Opened the Broswer. "
                    dbusMethods.appOpenCloseCommands("browser", True, 3)

                elif _ in [
                    "enlarge the browser",
                    "open browser in enlarge mode",
                    "pop out the browser",
                ]:
                    # NEW_TEXT += "Sure, enlarged the browser. "
                    dbusMethods.appOpenCloseCommands("browser", True, 2)

                elif _ in ["minimize the browser", "pop in the browser"]:
                    # NEW_TEXT += "Sure, minimized the browser . "
                    dbusMethods.appOpenCloseCommands("browser", True, 1)

                elif _ == "close browser":
                    dbusMethods.appOpenCloseCommands("browser", False, 5)

                elif _ == "hide browser":
                    dbusMethods.appOpenCloseCommands("browser", False, 4)

                elif _ == "open training videos":
                    # NEW_TEXT += "Sure, Opened the training videos. "
                    dbusMethods.appOpenCloseCommands("trainingvideo", True, 3)

                elif _ in [
                    "enlarge the training videos",
                    "open training videos in enlarge mode",
                    "pop out the training videos",
                ]:
                    # NEW_TEXT += "Sure, enlarged the training videos. "
                    dbusMethods.appOpenCloseCommands("trainingvideo", True, 2)

                elif _ in [
                    "minimize the training videos",
                    "pop in the training videos",
                ]:
                    # NEW_TEXT += "Sure, minimized the training videos . "
                    dbusMethods.appOpenCloseCommands("trainingvideo", True, 1)

                elif _ == "close training videos":
                    dbusMethods.appOpenCloseCommands("trainingvideo", False, 5)

                elif _ == "hide training videos":
                    dbusMethods.appOpenCloseCommands("trainingvideo", False, 4)

                elif _ == "open dicom viewer":
                    # NEW_TEXT += "Sure, Opened dicom viewer. "
                    dbusMethods.appOpenCloseCommands("dicomviewer", True, 3)

                elif _ in [
                    "enlarge the dicom viewer",
                    "open dicom viewer in enlarge mode",
                    "pop out the dicom viewer",
                ]:
                    # NEW_TEXT += "Sure, enlarged the dicom viewer. "
                    dbusMethods.appOpenCloseCommands("dicomviewer", True, 2)

                elif _ in ["minimize the dicom viewer", "pop in the dicom viewer"]:
                    # NEW_TEXT += "Sure, minimized the dicom viewer . "
                    dbusMethods.appOpenCloseCommands("dicomviewer", True, 1)

                elif _ == "close dicom viewer":
                    dbusMethods.appOpenCloseCommands("dicomviewer", False, 5)

                elif _ == "hide dicom viewer":
                    dbusMethods.appOpenCloseCommands("dicomviewer", False, 4)

                elif _ in [
                    "please exit",
                    "stop speaking",
                    "please stop",
                    "please stop / stop",
                    "stop",
                ]:
                    dbusMethods.stopSpeakFunctionality()

                elif _.find("start timer where") != -1:
                    _json_str = _.split("start timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(
                        _json_str.strip().replace(";", ",").replace("'", '"')
                    )
                    print("hitting the dbus", flush=True)
                    dbusMethods.newTimerCommands(
                        command="Timer_start",
                        timer_value=_json["timer_value"].strip().split(":")
                        or ["00", "00", "00"],
                        label=_json.get("timer_name") or "",
                        color=_json.get("color") or "",
                    )

                elif _.find("start all timers where") != -1:
                    _json_str = _.split("start all timers where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(
                        _json_str.strip().replace(";", ",").replace("'", '"')
                    )
                    for _ in range(3):
                        print("hitting the dbus", flush=True)
                        dbusMethods.newTimerCommands(
                            command="Timer_start",
                            timer_value=_json["timer_value"].strip().split(":")
                            or ["00", "00", "00"],
                            label="",
                            color="",
                        )

                elif _.find("cancel timer where") != -1:
                    _json_str = _.split("cancel timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_delete",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("clear timer where") != -1:
                    _json_str = _.split("clear timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_delete",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("delete timer where") != -1:
                    _json_str = _.split("delete timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_delete",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("delete / cancel / clear timer where") != -1:
                    _json_str = _.split("delete / cancel / clear timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_delete",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("pause timer where") != -1:
                    _json_str = _.split("pause timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_stop",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("stop timer where") != -1:
                    _json_str = _.split("stop timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_stop",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("pause / stop timer where") != -1:
                    _json_str = _.split("pause / stop timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_stop",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("resume timer where") != -1:
                    _json_str = _.split("resume timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_resume",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("reset timer where") != -1:
                    _json_str = _.split("reset timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_reset",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("show timer where") != -1:
                    _json_str = _.split("show timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_show",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "hide the timer":
                    dbusMethods.newTimerCommands(
                        command="Timer_hide",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "delete the timer":
                    dbusMethods.newTimerCommands(
                        command="Timer_delete",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("hide timer where") != -1:
                    _json_str = _.split("hide timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_hide",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "cancel all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_delete_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "delete all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_delete_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "clear all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_delete_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "delete / cancel / clear all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_delete_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _.find("how much time left for timer where") != -1:
                    _json_str = _.split("how much time left for timer where")[1]
                    print("generating json...", flush=True)
                    _json = json.loads(_json_str.strip().replace("'", '"'))
                    dbusMethods.newTimerCommands(
                        command="Timer_remaining",
                        label=_json.get("timer_name") or "",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "how much time left on all timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_running_list",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ in ["what timers are set", "how many timers are set"]:
                    dbusMethods.newTimerCommands(
                        command="Timer_running_list",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "display all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_show_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "hide all the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_hide_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "reset the timers":
                    dbusMethods.newTimerCommands(
                        command="Timer_reset_all",
                        label="",
                        color="",
                        timer_value=["00", "00", "00"],
                    )

                elif _ == "open system settings":
                    print("Command: open system settings")

                elif _.find("open preference card where") != -1:
                    _res = _.split("open preference card where")[1].strip()

                    data = json.loads(_res.replace(";", ",").replace("'", '"'))
                    payload = {"payload": "prefcard/command", "data": data}
                    dbusMethods.prefCardCommands(payload=payload)

                elif _ == "collect diagnostic logs":
                    payload = {
                        "payload": "collectlogs/command",
                        "command": "Collect_logs",
                        "capture_to_usb": 0,
                    }
                    dbusMethods.collectDiagnosticLogsCommands(
                        payload=json.dumps(payload)
                    )

                elif _ == "collect diagnostic logs on usb":
                    payload = {
                        "payload": "collectlogs/command",
                        "command": "Collect_logs",
                        "capture_to_usb": 1,
                    }
                    dbusMethods.collectDiagnosticLogsCommands(
                        payload=json.dumps(payload)
                    )

                elif _ in [
                    "turn tourniquet on",
                    "turn tourniquet up",
                    "start tourniquet",
                    "resume tourniquet",
                    "start tourniquet with force as false",
                ]:
                    dbusMethods.tourniquetCommands(
                        command="Start_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _ in [
                    "turn tourniquet off",
                    "turn tourniquet down",
                    "stop tourniquet",
                    "pause tourniquet",
                ]:
                    dbusMethods.tourniquetCommands(
                        command="Stop_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _ in ["reset tourniquet", "delete tourniquet"]:
                    dbusMethods.tourniquetCommands(
                        command="Reset_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _ == "show tourniquet":
                    dbusMethods.tourniquetCommands(
                        command="Show_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _ == "hide tourniquet":
                    dbusMethods.tourniquetCommands(
                        command="Hide_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _ in ["show tourniquet duration in active state", "how long the tourniquet has been up"]:
                    dbusMethods.tourniquetCommands(
                        command="Duration_tourniquet",
                        force=False,
                        time_value=["00", "00", "00"],
                    )

                elif _.find("start tourniquet where") != -1:
                    _res = _.split("start tourniquet where")[1].strip()

                    data = json.loads(_res.replace(";", ",").replace("'", '"'))
                    dbusMethods.tourniquetCommands(
                        command="Start_tourniquet",
                        force=False,
                        time_value=data["time_value"].strip().split(":")
                        or ["00", "00", "00"],
                    )

                elif _.find("start tourniquet with") != -1:
                    _res = _.split("start tourniquet with")[1].strip()

                    _time_value = _res.split(":")
                    dbusMethods.tourniquetCommands(
                        command="Start_tourniquet", force=True, time_value=_time_value
                    )

                elif _.find("extend tourniquet with") != -1:
                    _res = _.split("extend tourniquet with")[1].strip()

                    _time_value = _res.split(":")
                    dbusMethods.tourniquetCommands(
                        command="Addon_tourniquet", force=False, time_value=_time_value
                    )

                elif _ == "show current time":
                    print(
                        "Dbus to hit for getting the current time and speaking it",
                        flush=True,
                    )
                    _payload = {
                        "payload": "general/command",
                        "data": {"command": "Time_Format"},
                    }
                    try:
                        _response = dbusMethods.executeDBus(
                            command=json.dumps(_payload)
                        )
                        _res_json = json.loads(_response)
                        _time_format: int = int(_res_json["data"]["timeFormat"])
                    except Exception as e:
                        print(e)
                        print(
                            "Failed to fetch the system time format, defaults to 24 hours format..",
                            flush=True,
                        )
                        _time_format: int = 24

                    speakResponse.timeSpeakResponse(time_format=_time_format)

                elif _ in [
                    "enlarge the zoom app",
                    "pop out the zoom app",
                    "open zoom app in enlarge mode",
                ]:
                    dbusMethods.appOpenCloseCommands("zoom", True, 2)
                    # print("Hitting the dbus for enlarging the zoom app", flush=True)

                elif _ in ["minimize the zoom app", "pop in the zoom app"]:
                    # print("Hitting the dbus for minimizing the zoom app", flush=True)
                    dbusMethods.appOpenCloseCommands("zoom", True, 1)

                elif _ in ["open ai device help", "help with a device"]:
                    print("Executing the dbus for opening ai device help", flush=True)
                    dbusMethods.appOpenCloseCommands("aidevicehelp", True, 3)

                elif _ == "close ai device help":
                    print("Executing the dbus for closing ai device help", flush=True)
                    dbusMethods.appOpenCloseCommands("aidevicehelp", False, 5)

                elif _ == "open nvidia emr chatbot":
                    print(
                        "Executing the dbus for opening nvidia emr chatbot", flush=True
                    )
                    dbusMethods.appOpenCloseCommands("emrchatbot", True, 3)

                elif _ == "close nvidia emr chatbot":
                    print(
                        "Executing the dbus for closing nvidia emr chatbot", flush=True
                    )
                    dbusMethods.appOpenCloseCommands("emrchatbot", False, 5)

                elif _.find("enable / turn on ai solution where") != -1:
                    _res = _.split("enable / turn on ai solution where")[1].strip()

                    _data = json.loads(_res.replace(";", ",").replace("'", '"'))

                    _data["color"] = _data.get("color") or "white"
                    _data["opacity"] = _data.get("opacity") or "100"
                    _web_url: str = ""
                    try:
                        # Hitting the api for the app
                        _web_url = dbusMethods.aiAppApiCalling(
                            model=_data.get("name"),
                            color=_data.get("color"),
                            enable=True,
                            opacity=_data.get("opacity"),
                        )
                        # Hitting the dbus for opening the app
                        if _web_url != "":
                            # dbusMethods.appOpenCloseCommands(
                            #     "aidashboard", True, 3, _web_url
                            # )
                            # print(
                            #     "Hitting the dbus for enabling the ai solution apps",
                            #     flush=True,
                            # )
                            print("The data is: ", _data, flush=True)
                    except Exception as e:
                        print(e, flush=True)

                elif _.find("disable / turn off ai solution where") != -1:
                    _res = _.split("disable / turn off ai solution where")[1].strip()

                    _data = json.loads(_res.replace(";", ",").replace("'", '"'))
                    _web_url: str = ""
                    try:
                        # Hitting the api for the app
                        _web_url = dbusMethods.aiAppApiCalling(
                            model=_data.get("name"), enable=False
                        )

                        # Hitting the dbus for closing the app
                        if _web_url != "":
                            # dbusMethods.appOpenCloseCommands("aidashboard", False, 5)
                            # print(
                            #     "Hitting the dbus for disabling the ai solution apps",
                            #     flush=True,
                            # )
                            print("The data is: ", _data, flush=True)
                    except Exception as e:
                        print(e, flush=True)

                elif _ in [
                    "start / open ai copilot",
                    "start ai copilot",
                    "open ai copilot",
                ]:
                    print("Executing the dbus for opening ai copilot", flush=True)
                    dbusMethods.appOpenCloseCommands("aicopilot", True, 3)

                elif _ == "close ai copilot":
                    print("Executing the dbus for closing ai copilot", flush=True)
                    dbusMethods.appOpenCloseCommands("aicopilot", False, 5)

                elif _ == "TOURNIQUET_DISABLED":
                    speakResponse.appDisabledSpeakList.append(
                        "Tourniquet is currently disabled, please enable it from preferences to proceed."
                    )

                elif _ == "ZOOM_DISABLED":
                    speakResponse.appDisabledSpeakList.append(
                        "Zoom is currently disabled, please enable it from preferences to proceed."
                    )

                elif _.find("run command for patient where") != -1:
                    _res = _.split("run command for patient where")[1].strip()
                    _data = json.loads(_res.replace("'", '"'))

                    _action = _data.get("action")
                    _patient_actions = {
                        "in": ("patient_in", "NULL", "NULL"),
                        "out": ("patient_out", "NULL", "NULL"),
                    }

                    _params = _patient_actions.get(_action)

                    if _params is None:
                        speakResponse.speakList.append(
                            f"There is no such command for patient"
                        )
                    else:
                        dbusMethods.surgeryCommand(*_params)

                else:
                    # NEW_TEXT += _
                    print("NO MATCH FOUND...")
                    generalMethods.speakMessageWithBlocker(
                        f"Sorry, There is no such command {_}"
                    )
            except Exception as e:
                print(e)

        return speakResponse.speakList, speakResponse.appDisabledSpeakList
