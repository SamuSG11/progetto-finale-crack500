from it.akron.dataset import CrackDataset


def main():

    # 👉 IMPORTANTISSIMO: usa keyword arguments (evita bug tipo il tuo)
    ds = CrackDataset(split="train")

    print("\n--- DATASET INFO ---")
    print("Samples:", len(ds))

    # safety check
    if len(ds) == 0:
        print("Dataset vuoto: controlla path")
        return

    # test primo elemento
    img, mask = ds[0]

    print("\n--- SAMPLE SHAPE ---")
    print("Image shape:", img.shape)
    print("Mask shape:", mask.shape)

    # sanity check valori
    print("\n--- VALUE CHECK ---")
    print("Image min/max:", img.min(), img.max())
    print("Mask unique values:", set(mask.flatten()))


if __name__ == "__main__":
    main()