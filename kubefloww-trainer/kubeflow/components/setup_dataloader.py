import os
from kfp.dsl import Input, Output, component, Artifact

from .constants import KubeflowConfiguration


@component(
    base_image=KubeflowConfiguration.BASE_IMAGE,
    output_component_file=os.path.join(
        KubeflowConfiguration.ARRTIFACT_DIRECTORY, "components", "generate-dataset.yaml"
    ),
    packages_to_install=[
        "transformers",
        "lightning",
        "torchmetrics",
        "scikit-learn",
        "black",
        "datasets",
        "torchserve",
        "torch-model-archiver",
    ],
)
def setup_dataloaders(
    dataset: Input[Artifact],
):
    import os
    import torch
    from datasets import load_dataset
    from torch.utils.data import Dataset, DataLoader

    class IMDBDataset(Dataset):
        def __init__(self, dataset_dict, partition_key="train"):
            self.partition = dataset_dict[partition_key]

        def __getitem__(self, index):
            return self.partition[index]

        def __len__(self):
            return self.partition.num_rows

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    imdb_tokenized = load_dataset(dataset.path)

    train_dataset = IMDBDataset(imdb_tokenized, partition_key="train")
    val_dataset = IMDBDataset(imdb_tokenized, partition_key="validation")
    test_dataset = IMDBDataset(imdb_tokenized, partition_key="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=12,
        num_workers=1,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=12,
        num_workers=1,
        drop_last=True,
    )
