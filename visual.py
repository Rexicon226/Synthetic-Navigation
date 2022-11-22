import subprocess
import sys

import PySimpleGUI as sg


def main():
    sg.theme('DarkAmber')
    column_to_be_centered = [[sg.InputText(size=4, key='rows'), sg.Text('Rows')],
                             [sg.InputText(size=4, key='col'), sg.Text('Columns')],
                             [sg.InputText(size=4, key='oct'), sg.Text('Octaves')],
                             [sg.Checkbox(text="Show Progress Bar", key='prog', checkbox_color='White')],
                             [sg.InputText(size=4, key='seed'), sg.Text('Seed')], ]
    layout = [[sg.Column(column_to_be_centered, vertical_alignment='center', justification='center', k='-C-')],
              [sg.Output(size=(60, 7))],
              [sg.Button('Run'), sg.Button('Exit')]]  # a couple of buttons

    window = sg.Window('Command Output', layout)

    while True:  # Event Loop
        event, values = window.Read()
        if event in (None, 'Exit'):  # checks if user wants to exit
            break

        if event == 'Run':
            print('py -c "import blobcheck; blobcheck.visualize(' + str(values['rows']) + ', '
                  + str(values['col']) + ', ' + str(values['oct']) + ', '
                  + str(values['prog']) + ', ' + str(values['seed']) + ')"')
            runCommand(cmd='py -c "import blobcheck; blobcheck.visualize(' + str(values['rows']) + ', '
                           + str(values['col']) + ', ' + str(values['oct']) + ', '
                           + str(values['prog']) + ', ' + str(values['seed']) + ')"', window=window)

    window.Close()


# This function does the actual "running" of the command.  Also watches for any output. If found output is printed
def runCommand(cmd, timeout=None, window=None):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = ''
    for line in p.stdout:
        line = line.decode(errors='replace' if (sys.version_info) < (3, 5) else 'backslashreplace').rstrip()
        output += line
        print(line)
        window.Refresh() if window else None  # yes, a 1-line if, so shoot me
    retval = p.wait(timeout)
    return (retval, output)  # also return the output just for fun


if __name__ == '__main__':
    main()
