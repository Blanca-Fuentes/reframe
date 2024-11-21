import reframe.core.config as config

site_configuration = config.detect_config(
    exclude_feats=[],
    # exclude_feats=[],
    detect_containers=True,
    sched_options=[],
    time_limit=200,
    filename='system_config'
)
