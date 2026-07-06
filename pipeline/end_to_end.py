import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from localisation.ast_detector import ASTInPlaceAPIMisuseDetector, detect_inplace_api_misuse
from localisation.unixcoder_model import UniXcoderConfig, UniXcoderLocaliser
from refactoring.codet5_model import CodeT5Config, CodeT5Refactor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end localisation and refactoring on a Python snippet.")
    parser.add_argument("--input-file", required=True, help="Path to the Python source file.")
    parser.add_argument("--smell-type", default="In-Place APIs Misused", help="Smell type for conditioning.")
    parser.add_argument("--use-ast", action="store_true", help="Use the deterministic AST detector first.")
    parser.add_argument("--localiser-model", default="microsoft/unixcoder-base", help="Localiser model name or path.")
    parser.add_argument("--refactor-model", default="Salesforce/codet5-base", help="Refactor model name or path.")
    parser.add_argument("--max-localiser-length", type=int, default=256)
    parser.add_argument("--beam-size", type=int, default=5)
    args = parser.parse_args()

    source_path = Path(args.input_file)
    source_code = source_path.read_text(encoding="utf-8")

    if args.use_ast:
        detector = ASTInPlaceAPIMisuseDetector()
        localisation = detector.predict(source_code)
    else:
        config = UniXcoderConfig(model_name=args.localiser_model, max_length=args.max_localiser_length)
        detector = UniXcoderLocaliser(config=config)
        localisation = detector.predict(source_code)

    print("Localisation result:")
    print(localisation)

    refactor_config = CodeT5Config(model_name=args.refactor_model, beam_size=args.beam_size)
    refactorer = CodeT5Refactor(config=refactor_config)
    corrected = refactorer.generate(args.smell_type, source_code)

    print("Refactored output:")
    print(corrected)


if __name__ == "__main__":
    main()
