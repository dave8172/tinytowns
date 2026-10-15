// Types for tinytowns.js. The model is loaded from a .safetensors file and runs on your own machine.

declare namespace TinyTowns {
  type Size = "tinytowns-nano" | "tinytowns-small" | "tinytowns";

  interface Model {
    /** The model's settings, read from the weights file. */
    config: Record<string, number>;
    /** How many weights (numbers) the model has. */
    weights: number;
    /** 28 probabilities for the symbol after "." + prefix, in ALPHABET order. Index 0 is "end of name". */
    next(prefix: string, temperature?: number): number[];
    /** Keeps adding letters until the model ends the name, or it is 20 letters long. */
    sample(prefix?: string, options?: { temperature?: number; random?: () => number }): string;
  }
}

declare const TinyTowns: {
  /** ". abcdefghijklmnopqrstuvwxyz": "." is start and end of a name, then space, then a to z. */
  ALPHABET: string;
  /** 21: one "." plus at most 20 letters. */
  CONTEXT: number;
  /** Loads a model from a URL, or from the bytes of a .safetensors file. */
  load(source: string | ArrayBuffer): Promise<TinyTowns.Model>;
  /** The Hugging Face URL of a model's weights file. */
  url(size: TinyTowns.Size): string;
};

export = TinyTowns;
