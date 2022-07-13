from imgaug import augmenters as iaa

# ===============================================================================
# Aumentado de Datos - Severe
# ===============================================================================
def createSevereDataAugmentation():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order

    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
    ])
    """
    return seq



# ===============================================================================
# Aumentado de Datos - Mild
# ===============================================================================
def createMildDataAugmentation():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Affine(
            rotate=(-25, 25),
        )
    ], random_order=True) # apply augmenters in random order

    """
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
    ])
    """
    return seq
