import reframe.core.config as config

site_configuration = config.detect_config(
    exclude_feats=['c*-*', 'row*', 'contbuild', 'startx', 'group*', '128*'],
    # exclude_feats=[],
    detect_containers=True,
    sched_options=['-A csstaff'],
    time_limit=200,
    filename='system_config'
)
