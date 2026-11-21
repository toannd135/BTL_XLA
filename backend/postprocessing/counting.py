"""Count one authoritative instance list; never sum provider totals."""


def count_instances(instances):
    by_class = {name: 0 for name in ("banana", "apple", "orange")}
    for instance in instances:
        name = instance.class_name if hasattr(instance, "class_name") else instance.get("label")
        if name in by_class:
            by_class[name] += 1
    return {"total": sum(by_class.values()), "by_class": by_class,
            "source_instance_count": len(instances)}

