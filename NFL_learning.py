import numpy as np
import pandas as pd
import itertools
import random
from sklearn.metrics import accuracy_score

debug = False
CONF_THRE = 0.3
BASE_DIR = '../input/nfl-health-and-safety-helmet-assignment'

labels = pd.read_csv(f'{BASE_DIR}/train_labels.csv')
if debug:
    tracking = pd.read_csv(f'{BASE_DIR}/train_player_tracking.csv')
    helmets = pd.read_csv(f'{BASE_DIR}/train_baseline_helmets.csv')
else:
    tracking = pd.read_csv(f'{BASE_DIR}/test_player_tracking.csv')
    helmets = pd.read_csv(f'{BASE_DIR}/test_baseline_helmets.csv')


def find_nearest(array, value):
    # 输出输入数组与目标值差值的最小值
    value = int(value)
    array = np.asarray(array).astype(int)
    idx = (np.abs(array - value)).argmin()  # 数组展平后最小的那个值的索引
    return array[idx]


def norm_arr(a):
    # 输出归一化的数组
    a = a - a.min()
    a = a / a.max()
    return a


def dist(a1, a2):
    # 求输入两个矩阵差值矩阵的二范数 根号下的每个数的平方的和
    return np.linalg.norm(a1 - a2)


max_iter = 2000


def dist_for_different_len(a1, a2):
    """
    本函数保证返回相同长度的a1, a2 的二范数
    若长度不同则找到长度最小的二范数返回
    """
    # 如果a1长度小于a2则报错
    assert len(a1) >= len(a2), f'{len(a1)}, {len(a2)}'
    # a1 和 a2 的长度差值
    len_diff = len(a1) - len(a2)
    # 对a2 归一化
    a2 = norm_arr(a2)
    # 如果两者长度相同 返回 a1 a2 的二范数和一个()
    if len_diff == 0:
        a1 = norm_arr(a1)
        return dist(a1, a2), ()
    else:
        min_dist = 10000
        min_detete_idx = None
        cnt = 0
        # del_list 是 所有a1 中长度为两者差值的各种序号的列表
        del_list = list(itertools.combinations(range(len(a1)), len_diff))
        if len(del_list) > max_iter:
            # 限制在2000个之内
            del_list = random.sample(del_list, max_iter)
        for detete_idx in del_list:
            """
            detete_idx 是长度为两个数组差值的随机序号数组
            本段代码含义是每次随机删除差值个数的值
            归一化剩下的值
            求归一化后的数组的二范数
            求所有的二范数中最小的
            return: 最小的二范数，即两者的距离； 此时删除的序号
            """
            this_a1 = np.delete(a1, detete_idx)
            this_a1 = norm_arr(this_a1)
            this_dist = dist(this_a1, a2)
            # print(len(a1), len(a2), this_dist)
            if min_dist > this_dist:
                min_dist = this_dist
                min_detete_idx = detete_idx

        return min_dist, min_detete_idx


def rotate_arr(u, t, deg=True):
    if deg == True:
        # t 是将角度转换为弧度的数组array
        t = np.deg2rad(t)
    R = np.array([[np.cos(t), -np.sin(t)],
                  [np.sin(t), np.cos(t)]])
    return np.dot(R, u)


def dist_rot(tracking_df, a2):
    """
    猜测：
    本函数逻辑
    首先得到轨迹的x,y值
    设置随机旋转角度
    求出旋转后的轨迹x与比较矩阵x的距离最小的那个角度的x值
    根据这个x值筛选出需要的player的id
    return: 最小距离, id
    """
    tracking_df = tracking_df.sort_values('x')
    x = tracking_df['x']
    y = tracking_df['y']
    min_dist = 10000
    min_idx = None
    min_x = None
    dig_step = 3
    dig_max = dig_step * 10
    # dig = -30, -27, …… , 27, 30
    for dig in range(-dig_max, dig_max + 1, dig_step):
        # dig 是随机旋转的角度，随机将(x,y)乘上角度
        arr = rotate_arr(np.array((x, y)), dig)
        # np.sort() 默认将数组按行排序
        this_dist, this_idx = dist_for_different_len(np.sort(arr[0]), a2)
        if min_dist > this_dist:
            min_dist = this_dist
            min_idx = this_idx
            min_x = arr[0]
    tracking_df['x_rot'] = min_x
    player_arr = tracking_df.sort_values('x_rot')['player'].values
    players = np.delete(player_arr, min_idx)
    return min_dist, players


def mapping_df(args):
    video_frame, df = args
    gameKey, playID, view, frame = video_frame.split('_')
    gameKey = int(gameKey)
    playID = int(playID)
    frame = int(frame)
    this_tracking = tracking[(tracking['gameKey'] == gameKey) & (tracking['playID'] == playID)]
    est_frame = find_nearest(this_tracking.est_frame.values, frame)
    this_tracking = this_tracking[this_tracking['est_frame'] == est_frame]
    len_this_tracking = len(this_tracking)
    df['center_h_p'] = (df['left'] + df['width'] / 2).astype(int)
    df['center_h_m'] = (df['left'] + df['width'] / 2).astype(int) * -1
    df = df[df['conf'] > CONF_THRE].copy()
    if len(df) > len_this_tracking:
        df = df.tail(len_this_tracking)
    df_p = df.sort_values('center_h_p').copy()
    df_m = df.sort_values('center_h_m').copy()
    # end zone:（球门线到底线的）球门区；结束区
    if view == 'Endzone':
        this_tracking['x'], this_tracking['y'] = this_tracking['y'].copy(), this_tracking['x'].copy()
    a2_p = df_p['center_h_p'].values
    a2_m = df_m['center_h_m'].values

    min_dist_p, min_detete_idx_p = dist_rot(this_tracking, a2_p)
    min_dist_m, min_detete_idx_m = dist_rot(this_tracking, a2_m)
    if min_dist_p < min_dist_m:
        min_dist = min_dist_p
        min_detete_idx = min_detete_idx_p
        tgt_df = df_p
    else:
        min_dist = min_dist_m
        min_detete_idx = min_detete_idx_m
        tgt_df = df_m
    # print(video_frame, len(this_tracking), len(df), len(df[df['conf']>CONF_THRE]), this_tracking['x'].mean(), min_dist_p, min_dist_m, min_dist)
    tgt_df['label'] = min_detete_idx
    return tgt_df[['video_frame', 'left', 'width', 'top', 'height', 'label']]


p = Pool(processes=4)
submission_df_list = []
df_list = list(helmets.groupby('video_frame'))
with tqdm(total=len(df_list)) as pbar:
    for this_df in p.imap(mapping_df, df_list):
        submission_df_list.append(this_df)
        pbar.update(1)
p.close()

submission_df = pd.concat(submission_df_list)
submission_df.to_csv('submission.csv', index=False)

# copied from https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide
class NFLAssignmentScorer:
    def __init__(
        self,
        labels_df: pd.DataFrame = None,
        labels_csv="train_labels.csv",
        check_constraints=True,
        weight_col="isDefinitiveImpact",
        impact_weight=1000,
        iou_threshold=0.35,
        remove_sideline=True,
    ):
        """
        Helper class for grading submissions in the
        2021 Kaggle Competition for helmet assignment.
        Version 1.0
        https://www.kaggle.com/robikscube/nfl-helmet-assignment-getting-started-guide

        Use:
        ```
        scorer = NFLAssignmentScorer(labels)
        scorer.score(submission_df)

        or

        scorer = NFLAssignmentScorer(labels_csv='labels.csv')
        scorer.score(submission_df)
        ```

        Args:
            labels_df (pd.DataFrame, optional):
                Dataframe containing theground truth label boxes.
            labels_csv (str, optional): CSV of the ground truth label.
            check_constraints (bool, optional): Tell the scorer if it
                should check the submission file to meet the competition
                constraints. Defaults to True.
            weight_col (str, optional):
                Column in the labels DataFrame used to applying the scoring
                weight.
            impact_weight (int, optional):
                The weight applied to impacts in the scoring metrics.
                Defaults to 1000.
            iou_threshold (float, optional):
                The minimum IoU allowed to correctly pair a ground truth box
                with a label. Defaults to 0.35.
            remove_sideline (bool, optional):
                Remove slideline players from the labels DataFrame
                before scoring.
        """
        if labels_df is None:
            # Read label from CSV
            if labels_csv is None:
                raise Exception("labels_df or labels_csv must be provided")
            else:
                self.labels_df = pd.read_csv(labels_csv)
        else:
            self.labels_df = labels_df.copy()
        if remove_sideline:
            self.labels_df = (
                self.labels_df.query("isSidelinePlayer == False")
                .reset_index(drop=True)
                .copy()
            )
        self.impact_weight = impact_weight
        self.check_constraints = check_constraints
        self.weight_col = weight_col
        self.iou_threshold = iou_threshold

    def check_submission(self, sub):
        """
        Checks that the submission meets all the requirements.

        1. No more than 22 Boxes per frame.
        2. Only one label prediction per video/frame
        3. No duplicate boxes per frame.

        Args:
            sub : submission dataframe.

        Returns:
            True -> Passed the tests
            False -> Failed the test
        """
        # Maximum of 22 boxes per frame.
        max_box_per_frame = sub.groupby(["video_frame"])["label"].count().max()
        if max_box_per_frame > 22:
            print("Has more than 22 boxes in a single frame")
            return False
        # Only one label allowed per frame.
        has_duplicate_labels = sub[["video_frame", "label"]].duplicated().any()
        if has_duplicate_labels:
            print("Has duplicate labels")
            return False
        # Check for unique boxes
        has_duplicate_boxes = (
            sub[["video_frame", "left", "width", "top", "height"]].duplicated().any()
        )
        if has_duplicate_boxes:
            print("Has duplicate boxes")
            return False
        return True

    def add_xy(self, df):
        """
        Adds `x1`, `x2`, `y1`, and `y2` columns necessary for computing IoU.

        Note - for pixel math, 0,0 is the top-left corner so box orientation
        defined as right and down (height)
        对于像素数学，0,0是左上角，所以是方框方向定义为右向下(高度)
        """

        df["x1"] = df["left"]
        df["x2"] = df["left"] + df["width"]
        df["y1"] = df["top"]
        df["y2"] = df["top"] + df["height"]
        return df

    def merge_sub_labels(self, sub, labels, weight_col="isDefinitiveImpact"):
        """
        Perform an outer join between submission and label.
        Creates a `sub_label` dataframe which stores the matched label for each submission box.
        Ground truth values are given the `_gt` suffix, submission values are given `_sub` suffix.
        """
        sub = sub.copy()
        labels = labels.copy()

        sub = self.add_xy(sub)
        labels = self.add_xy(labels)

        base_columns = [
            "label",
            "video_frame",
            "x1",
            "x2",
            "y1",
            "y2",
            "left",
            "width",
            "top",
            "height",
        ]
        """
        此处为两个dataframe的拼接
        左left :   sub[base_columns]
        右right:   labels[base_columns + [weight_col]]
        拼接方式: 按照两个的 video_frame 列进行拼接
        保留right 即label数组的 video_frame 列的索引
        增加其他值的_sub, _gt 列 
        每个列 分别保留其原来数组的值
        """
        sub_labels = sub[base_columns].merge(
            labels[base_columns + [weight_col]],
            on=["video_frame"],
            how="right",
            suffixes=("_sub", "_gt"),
        )
        return sub_labels

    def get_iou_df(self, df):
        """
        This function computes the IOU of submission (sub)
        bounding boxes against the ground truth boxes (gt).
        """
        df = df.copy()

        # 1. get the coordinate of inters
        df["ixmin"] = df[["x1_sub", "x1_gt"]].max(axis=1)
        df["ixmax"] = df[["x2_sub", "x2_gt"]].min(axis=1)
        df["iymin"] = df[["y1_sub", "y1_gt"]].max(axis=1)
        df["iymax"] = df[["y2_sub", "y2_gt"]].min(axis=1)

        df["iw"] = np.maximum(df["ixmax"] - df["ixmin"] + 1, 0.0)
        df["ih"] = np.maximum(df["iymax"] - df["iymin"] + 1, 0.0)

        # 2. calculate the area of inters
        df["inters"] = df["iw"] * df["ih"]

        # 3. calculate the area of union
        df["uni"] = (
            (df["x2_sub"] - df["x1_sub"] + 1) * (df["y2_sub"] - df["y1_sub"] + 1)
            + (df["x2_gt"] - df["x1_gt"] + 1) * (df["y2_gt"] - df["y1_gt"] + 1)
            - df["inters"]
        )
        # print(uni)
        # 4. calculate the overlaps between pred_box and gt_box
        df["iou"] = df["inters"] / df["uni"]

        return df.drop(
            ["ixmin", "ixmax", "iymin", "iymax", "iw", "ih", "inters", "uni"], axis=1
        )

    def filter_to_top_label_match(self, sub_labels):
        """
        Ensures ground truth boxes are only linked to the box
        in the submission file with the highest IoU.
        """
        return (
            # ;reset_index()重置索引
            sub_labels.sort_values("iou", ascending=False)
            .groupby(["video_frame", "label_gt"])
            .first()
            .reset_index()
        )

    def add_isCorrect_col(self, sub_labels):
        """
        Adds True/False column if the ground truth label
        and submission label are identical
        """
        sub_labels["isCorrect"] = (
            sub_labels["label_gt"] == sub_labels["label_sub"]
        ) & (sub_labels["iou"] >= self.iou_threshold)
        return sub_labels

    def calculate_metric_weighted(
        self, sub_labels, weight_col="isDefinitiveImpact", weight=1000
    ):
        """
        Calculates weighted accuracy score metric.
        """
        sub_labels["weight"] = sub_labels.apply(
            lambda x: weight if x[weight_col] else 1, axis=1
        )
        y_pred = sub_labels["isCorrect"].values
        y_true = np.ones_like(y_pred)
        weight = sub_labels["weight"]
        return accuracy_score(y_true, y_pred, sample_weight=weight)

    def score(self, sub, labels_df=None, drop_extra_cols=True):
        """
        Scores the submission file against the labels.

        Returns the evaluation metric score for the helmet
        assignment kaggle competition.

        If `check_constraints` is set to True, will return -999 if the
            submission fails one of the submission constraints.
        """
        if labels_df is None:
            labels_df = self.labels_df.copy()

        if self.check_constraints:
            if not self.check_submission(sub):
                return -999
        sub_labels = self.merge_sub_labels(sub, labels_df, self.weight_col)
        sub_labels = self.get_iou_df(sub_labels).copy()
        sub_labels = self.filter_to_top_label_match(sub_labels).copy()
        sub_labels = self.add_isCorrect_col(sub_labels)
        score = self.calculate_metric_weighted(
            sub_labels, self.weight_col, self.impact_weight
        )
        # Keep `sub_labels for review`
        if drop_extra_cols:
            drop_cols = [
                "x1_sub",
                "x2_sub",
                "y1_sub",
                "y2_sub",
                "x1_gt",
                "x2_gt",
                "y1_gt",
                "y2_gt",
            ]
            sub_labels = sub_labels.drop(drop_cols, axis=1)
        self.sub_labels = sub_labels
        return score

if debug:
    scorer = NFLAssignmentScorer(labels)
    baseline_score = scorer.score(submission_df)
    print(f"validation score {baseline_score:0.4f}") # this would be 0.33