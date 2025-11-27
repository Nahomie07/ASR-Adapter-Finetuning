# ASR-Adapter-Finetuning (Whisper-small)

Structure:
- src/: training, evaluation and adapter insertion scripts
- adapters/: trained adapter weights (output)
- base_model/: optional snapshot of base model weights
- docs/: challenge instructions 

**Quick start**
1. Install requirements:
   pip install -r requirements.txt
2. Run training (GPU recommended):
   bash run_train.sh
3. Evaluate:
   bash run_eval.sh

The challenge brief was into docs/ASR_Fellowship_Challenge_Instructions.pdf
