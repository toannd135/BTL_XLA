"""Small provider router; optional providers are registered only when enabled."""


class InferenceRouter:
    def __init__(self, providers, default_provider="local_yolo"):
        self.providers = dict(providers)
        if default_provider not in self.providers:
            raise ValueError(f"Inference provider {default_provider!r} is not configured")
        self.default_provider = default_provider

    def predict(self, image, provider=None):
        selected = provider or self.default_provider
        if selected not in self.providers:
            raise ValueError(f"Inference provider {selected!r} is not configured")
        return self.providers[selected].predict(image)

