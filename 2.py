def _video_part_from_cluster_dir(self):
    cluster_dir = os.path.abspath(self.config.person_clusters_dir)

    if os.path.basename(cluster_dir) != "person_clusters":
        return None, None

    video_dir = os.path.dirname(cluster_dir)

    return os.path.basename(video_dir), None


if self.config.one_shot_process_dir:
    video, part = self._video_part_from_cluster_dir()

    if video:
        return os.path.join(
            self.config.one_shot_process_dir,
            video,
            str(item["shot_key"]),
            f"id_{item['obj_id']}",
            "features",
            "face_feature",
            f"{int(item['frame_idx'])}.npy",
        )
