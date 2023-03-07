def get_data_cls(cls_name: str):
    if cls_name == "Data":
        from teco.data.data import Data

        return Data
    else:
        raise ValueError(f"Unknown data class: {cls_name}")
