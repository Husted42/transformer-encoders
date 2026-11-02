import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils.corpus import Corpus


def print_split_summary(split_name, dataset):
    print(f"\n{'=' * 80}")
    print(f"SPLIT: {split_name}")
    print(f"{'=' * 80}")
    print(f"Dataset shape: {dataset.num_rows} rows")
    print(f"Columns: {dataset.column_names}")

    if dataset.num_rows == 0:
        print("This split is empty.")
        return

    first_row = dataset[0]
    print("\nFirst row fields:")
    for key, value in first_row.items():
        print(f"  - {key}: {value}")

    print("\nHuman-readable preview:")
    print(f"  Question: {first_row.get('question', '<missing>')}")

    text_value = first_row.get('text')
    if isinstance(text_value, list):
        print(f"  Relevant text entries: {len(text_value)}")
        for i, item in enumerate(text_value[:3]):
            print(f"    [{i}] {item[:200]}...")
    else:
        print(f"  Text preview: {str(text_value)[:200]}...")

    print(f"  Query ID: {first_row.get('query_id', '<missing>')}")
    print(f"  Document IDs: {first_row.get('document_id', '<missing>')}")


def main():
    print("Loading FIQA corpus...")
    corpus = Corpus(dataset_name='fiqa', base_dir='.')

    print(f"Available splits: {list(corpus.ds.keys())}")

    for split_name in ['train', 'validation', 'test']:
        ds = corpus.ds.get(split_name)
        if ds is not None:
            print_split_summary(split_name, ds)
        else:
            print(f"\n{'=' * 80}\nSPLIT: {split_name}\n{'=' * 80}\nMissing in this dataset.\n")


if __name__ == '__main__':
    main()
