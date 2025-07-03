# import ms1mv3, webface4m, webface12m

def get_config(file):
    if file == "ms1mv3":
        from configs.ms1mv3 import cfg
    elif file == "webface4m":
        from configs.webface4m import cfg
    elif file == "webface8m":
        from configs.webface8m import cfg
    elif file == "webface12m":
        from configs.webface12m import cfg
    elif file == "convnextv2_ms1m_arcface":
        print("Hello")
        from configs.convnextv2_ms1m_arcface import cfg
    else:
        print("Config file loading failed.")
        return None
    
    # for key, value in vars(cfg).items():
    #     print(str(key) + ", " + str(value))
        
    return cfg()

if __name__ == "__main__":
    cfg = get_config("convnextv2_ms1m_arcface")
    
    for key in dir(cfg):
        if key.startswith("__"):
            continue
        print(key + ", " + str(getattr(cfg, key)))
        