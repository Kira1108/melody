# Melody

## Install and use
```bash
git clone [this repo]
pip install .
```

```python
from melody.pipelines import SegmentationPipeline

seg = SegmentationPipeline()

seg.run("asr_example.wav")

>> ['你好你好。', '今天你开心吗？你们公司有什么产品？', '你们公司是做啥的。']
```