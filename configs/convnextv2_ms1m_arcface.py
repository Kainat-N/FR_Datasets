class cfg:
    # Margin Base Softmax
    margin_list = (1.0, 0.5, 0.0)

    # Partial FC

    sample_rate = 1.0
    interclass_filtering_threshold = 0


    fp16 = True

    optimizer = "sgd"
    lr = 0.02
    momentum = 0.9
    weight_decay = 5e-4

    verbose = 16000
    frequent = 10

    gradient_acc = 1
    seed = 3407

    dali = False
    dali_aug = False

    num_workers = 4
    batch_size = 128
    embedding_size = 512
    image_size = (112, 112)

    # 👇 Your custom model registered under this name (you’ll register convnextv2_base)
    network = "convnextv2_base"

    resume = False
    save_all_states = False

    # 👇 Path to your RecordIO dataset - training dataset
    rec = "E:/FAST/FR/datasets/ms1m_arcface_images"
    # Validation datset
    val = "E:/FAST/FR/datasets/lfw/lfw-deepfunneled"  # Optional: any val dataset (e.g. LFW)

    output = "output/convnextv2_ms1m"
    num_classes =85561
    num_image = 5822653
    num_epoch = 10  # you can increase this later
    steps_per_epoch = num_image // batch_size
    total_step = steps_per_epoch * num_epoch
    warmup_epoch = 0

    # val_targets = ["lfw", "cfp_fp", "agedb_30"]  # add/remove as needed
    val_targets = ["lfw"] 


