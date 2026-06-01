from DataGeneration.DataGenerator import DataGenerator
from ModelEvaluation.ModelEvaluator import ModelEvaluator


def main():
    print("=== KROK 1: GENEROWANIE ZBIORU DANYCH ===")
    generator = DataGenerator(root_dir="Dataset", split_ratio=0.8)
    generator.build_dataset(number_of_images=100, max_lines=10)

    print("\n=== KROK 2: UCZENIE I EWALUACJA MODELU ===")
    evaluator = ModelEvaluator(dataset_dir="Dataset")

    # evaluator.train_model(epochs=30, imgsz=640)

    evaluator.evaluate_line_count_accuracy()


if __name__ == "__main__":
    main()
