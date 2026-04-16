from pathlib import Path
from classifiers.raw_data_classifier import *

def main():
    # read shit from file
    operation = input("To prepare data inout 'p', to classify input 'c': ")

    # temp_path = Path("./classifiers/classifiers_data")
    # for file in temp_path.iterdir():
    #     if file.suffix != ".jsonl":
    #         continue

    #     timestamps, adc_outputs = load_segment_file(file)

    #     plt.figure(figsize=(10, 5))
    #     for channel_idx in range(adc_outputs.shape[1]):
    #         plt.plot(timestamps, adc_outputs[:, channel_idx], label=f"ADC {channel_idx + 1}")

    #     plt.title(file.name)
    #     plt.xlabel("Timestamp")
    #     plt.ylabel("ADC output")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()


    if operation == 'p':
        raw_data_dir = Path("./data/raw")

        for file in raw_data_dir.iterdir():
            print(file.name)
            timestamps, adc_data = process_raw_file(file)
            split_data_into_segments(file, adc_data, timestamps)
    else:
        model, test_loader, label_to_id, id_to_label = lstm_classification(Path("./classifiers/classifiers_data"))
        results_and_plot(model, test_loader, label_to_id, id_to_label)

    # lstm classification
    # plot and profit


if __name__ == "__main__":
    main()