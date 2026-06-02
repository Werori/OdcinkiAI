import glob
import os

from ultralytics.models.yolo import YOLO


class ModelEvaluator:
    def __init__(self, dataset_dir: str = "Dataset", model_name: str = "yolov8n.pt"):
        self.dataset_dir = dataset_dir
        self.model = YOLO(model_name)
        self.yaml_path = "dataset.yaml"

    def _create_yaml_config(self):
        abs_dataset_path = os.path.abspath(self.dataset_dir)

        yaml_content = (
            f"path: {abs_dataset_path}\n"
            f"train: TrainingData/images\n"
            f"val: ValidationData/images\n\n"
            f"names:\n"
            f"  0: linia\n"
        )

        with open(self.yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"Utworzono plik konfiguracyjny: {self.yaml_path}")

    def train_model(self, epochs: int = 10, imgsz: int = 640):
        self._create_yaml_config()

        print(f"Rozpoczynam uczenie modelu przez {epochs} epok...")
        self.model.train(data=self.yaml_path, epochs=epochs, imgsz=imgsz, device="cpu")
        print("Trening zakończony!")

    def evaluate_line_count_accuracy(self):
        trained_model_path = os.path.join(
            "runs", "detect", "train", "weights", "best.pt"
        )

        if os.path.exists(trained_model_path):
            print(f"Ładuję Twój wyuczony model z: {trained_model_path}")
            self.model = YOLO(trained_model_path)
        else:
            print(
                f"Ostrzeżenie: Nie znaleziono {trained_model_path}. Testuję na domyślnym modelu!"
            )
        print("\n--- ROZPOCZĘCIE WALIDACJI ---")

        val_images_route = os.path.join(
            self.dataset_dir, "ValidationData", "images", "*.png"
        )
        image_paths = glob.glob(val_images_route)

        if not image_paths:
            print(
                "Błąd: Nie znaleziono żadnych zdjęć w folderze ValidationData/images!"
            )
            return

        correct_predictions = 0
        total_images = len(image_paths)

        for img_path in image_paths:
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            true_line_count = int(name_without_ext.split("_")[1])

            results = self.model(img_path, conf=0.30, verbose=False)

            predicted_line_count = len(results[0].boxes)

            if true_line_count == predicted_line_count:
                correct_predictions += 1
                status = "OK"
            else:
                status = "BLAD"

            print(
                f"Obraz: {filename} | Prawdziwe: {true_line_count} | Wykryte: {predicted_line_count} -> [{status}]"
            )

        global_accuracy = (correct_predictions / total_images) * 100

        print("\n" + "=" * 60)
        print(f"GLOBALNA DOKŁADNOŚĆ DETEKCJI ODCINKÓW: {global_accuracy:.2f}%")
        print(
            f"Wymaganie projektowe (>= 90%): {'SPEŁNIONE' if global_accuracy >= 90 else 'NIE SPEŁNIONE'}"
        )
        print("=" * 60)
