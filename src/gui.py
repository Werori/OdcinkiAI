from tkinter import filedialog

import customtkinter as ctk

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class YOLOApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.setup_main_window()
        self.create_sidebar()
        self.create_main_panel()
        self.create_status_bar()

    def setup_main_window(self):
        self.title("Detektor Odcinków")
        self.geometry("1000x650")

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.sidebar, text="MENU", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="Wczytaj Obraz", command=self.load_image)
        self.btn_load.pack(padx=20, pady=10)

        self.conf_label = ctk.CTkLabel(self.sidebar, text="Próg Pewności (Conf): 0.30")
        self.conf_label.pack(padx=20, pady=(20, 0))
        
        # Bezpieczne liczby całkowite 5-100
        self.slider_conf = ctk.CTkSlider(self.sidebar, from_=5, to=100, command=self.update_conf)
        self.slider_conf.set(30)
        self.slider_conf.pack(padx=20, pady=5)

        self.btn_detect = ctk.CTkButton(self.sidebar, text="URUCHOM DETEKCJĘ", fg_color="green", hover_color="darkgreen", command=self.run_detection)
        self.btn_detect.pack(padx=20, pady=30)

    def create_main_panel(self):
        self.main_panel = ctk.CTkFrame(self)
        self.main_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.img_container = ctk.CTkFrame(self.main_panel, fg_color="transparent")
        self.img_container.pack(fill="both", expand=True, padx=20, pady=20)

        self.lbl_orig = ctk.CTkLabel(self.img_container, text="[ Podgląd Oryginału ]", width=350, height=350, fg_color="#2B2B2B", corner_radius=8)
        self.lbl_orig.pack(side="left", fill="both", expand=True, padx=10)

        self.lbl_res = ctk.CTkLabel(self.img_container, text="[ Wynik Detekcji YOLO ]", width=350, height=350, fg_color="#2B2B2B", corner_radius=8)
        self.lbl_res.pack(side="right", fill="both", expand=True, padx=10)

    def create_status_bar(self):
        self.status_panel = ctk.CTkFrame(self.main_panel, height=80)
        self.status_panel.pack(fill="x", side="bottom", padx=20, pady=10)

        self.lbl_counter = ctk.CTkLabel(self.status_panel, text="Wykryto odcinków: --", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_counter.pack(side="left", padx=20, pady=10)

        self.lbl_time = ctk.CTkLabel(self.status_panel, text="Czas operacji: -- ms", font=ctk.CTkFont(size=14))
        self.lbl_time.pack(side="right", padx=20, pady=10)

    def update_conf(self, value):
        conf_float = float(value) / 100.0
        self.conf_label.configure(text=f"Próg Pewności (Conf): {conf_float:.2f}")

    def load_image(self):
        pass

    def run_detection(self):
        pass

if __name__ == "__main__":
    app = YOLOApp()
    app.mainloop()
