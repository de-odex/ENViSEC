from src.utility import load_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from src.normalize_data import normalize_data


def get_single_val_cols(df):
    single_value_cols = []
    length = len(df)
    for col in df.columns:
        if df[col].nunique() == length or df[col].nunique() == 1:
            single_value_cols.append(col)
    return single_value_cols


edge_dnn = "./data/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"
df_dnn = pd.read_csv(edge_dnn)


def drop_data(df_raw):
    """
    Dropping data (Columns, duplicated rows, NAN, Null..):
    based on https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications
    """
    drop_columns = [
        "frame.time",
        "ip.src_host",
        "ip.dst_host",
        "arp.src.proto_ipv4",
        "arp.dst.proto_ipv4",
        "http.file_data",
        "http.request.full_uri",
        "icmp.transmit_timestamp",
        "http.request.uri.query",
        "tcp.options",
        "tcp.payload",
        "tcp.srcport",
        "tcp.dstport",
        "udp.port",
        "mqtt.msg",
    ]

    df = df_raw.drop(columns=drop_columns, axis=1)
    df = df.dropna(axis=0, how="any")
    df = df.drop_duplicates(subset=None, keep="first")
    df = shuffle(df)
    nan_dict = dict(df.isna().sum())
    nan_value_cols = [(x, nan_dict[x]) for x in nan_dict if nan_dict[x] > 0]
    if nan_value_cols:
        print("Following columns has nan values: Tuple(col, num_nan_values):-")
        print(nan_value_cols)
    return df


def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


df = drop_data(df_dnn)
encode_text_dummy(df, "http.request.method")
encode_text_dummy(df, "http.referer")
encode_text_dummy(df, "http.request.version")
encode_text_dummy(df, "dns.qry.name.len")
encode_text_dummy(df, "mqtt.conack.flags")
encode_text_dummy(df, "mqtt.protoname")
encode_text_dummy(df, "mqtt.topic")
df = df.reset_index(drop=True)

if "index" in df.columns:
    df = df.drop(columns=["index"], axis=1)
df = df.sample(frac=1).reset_index(drop=True)


# print(len(df_final.drop(columns=['multi_label']).drop_duplicates()
def remove_dupsX(df):
    """
    remove the duplicate rows conflicting with y classes
    """
    independent_features = [col for col in list(df.columns) if col != "multi_label"]
    return df.drop_duplicates(subset=independent_features, keep=False).reset_index(
        drop=False
    )


if "Attack_type" in df.columns:
    df = df.rename(columns={"Attack_type": "multi_label"})
print(len(df))
df = remove_dupsX(df)
print(len(df))

df["multi_label"] = df["multi_label"].str.replace("Normal", "benign")
df_norm = normalize_data(df.drop(columns=["multi_label"], axis=1))
df_norm["multi_label"] = df["multi_label"]
df_norm.to_csv("./data/normalized/EdgeIIoTset.csv", index=False)
