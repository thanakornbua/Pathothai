class ChannelControls:
    def __init__(self, master, image_data):
        self.master = master
        self.image_data = image_data
        self.channel_vars = {
            'R': tk.BooleanVar(value=True),
            'G': tk.BooleanVar(value=True),
            'B': tk.BooleanVar(value=True)
        }
        self.create_widgets()

    def create_widgets(self):
        channel_frame = tk.Frame(self.master)
        channel_frame.pack(side=tk.TOP, fill=tk.X)

        for channel, var in self.channel_vars.items():
            chk = tk.Checkbutton(channel_frame, text=channel, variable=var, command=self.update_image)
            chk.pack(side=tk.LEFT)

    def update_image(self):
        # Logic to update the displayed image based on the selected channels
        pass

    def get_selected_channels(self):
        return [channel for channel, var in self.channel_vars.items() if var.get()]