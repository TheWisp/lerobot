"""Version of computed FLASH results, independent of file/plan schemas.

Increment for changes to frame processing, grouping, association, pose fitting,
fusion or persisted display evidence. Settings are compared separately. Missing
versions are legacy results, never evidence that the current code produced them.
"""

PIPELINE_VERSION = 1


def is_current(result):
    return type(result.get("pipeline_version")) is int and result["pipeline_version"] == PIPELINE_VERSION
