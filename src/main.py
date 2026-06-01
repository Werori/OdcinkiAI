from DataGeneration.DataGenerator import DataGenerator


def main():
    generator = DataGenerator(root_dir="Dataset", split_ratio=0.8)
    generator.build_dataset(number_of_images=1000, max_lines=10)

if __name__ == "__main__":
    main()
