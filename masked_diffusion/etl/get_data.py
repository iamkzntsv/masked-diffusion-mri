from datasets import load_dataset


def download_and_save_dataset(dataset_name, save_path) -> None:
    # Load the dataset from HuggingFace hub
    dataset = load_dataset(dataset_name)

    # Save the dataset locally
    dataset.save_to_disk(save_path)


if __name__ == "__main__":
    dataset_name = "iamkzntsv/IXI2D"

    save_path = "data/ixi/transformed"

    download_and_save_dataset(dataset_name, save_path)
