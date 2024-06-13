# import the library
from appJar import gui

# handle button events
def press(button):
    if button == "Cancel":
        app.stop()
    else:
        usr = app.getEntry("Entry")
        print("User:", usr)

# create a GUI variable called app
app = gui("Personality Prediction", "400x200")
app.setBg("black")
app.setFont(18)


# add & configure widgets - widgets get a name, to help referencing them later
app.addFlashLabel("title", "Welcome to the AI Personality Predictor")
app.setLabelBg("title", "gray")
app.setLabelFg("title", "white")
app.setLabelFont("title",62)

app.addLabelEntry("Entry")
app.setLabelBg("Entry","gray")
app.setLabelFg("Entry","white")


# link the buttons to the function called press
app.addButtons(["Submit", "Cancel"], press)

app.setFocus("Entry")

# start the GUI
app.go()