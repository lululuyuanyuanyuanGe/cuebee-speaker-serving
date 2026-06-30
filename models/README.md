# Model mount point

The expected production artifact is
`3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx`. Model binaries are excluded from
Git. Mount or copy the verified artifact here and pass its path with both `--model` and the
compiled worker path:

```bash
cuebee-speaker-server \
  --native-worker build/native-onnx/cpp/cuebee-speaker-worker \
  --native-backend onnx \
  --model models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
```

The service validates the observed output dimension at runtime. Artifact provenance and
the complete SHA-256 (Secure Hash Algorithm 256-bit) digest should be recorded by the
deployment system; this repository does not invent the truncated digest from the project
brief.
