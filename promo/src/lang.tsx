import { createContext, useContext } from "react";

/**
 * "bilingual" keeps the Chinese lines (the series' Chinese name and a few captions);
 * "en" is the English-only cut (for X and other English-language channels).
 */
export type Lang = "bilingual" | "en";
export const LangContext = createContext<Lang>("bilingual");
export const useLang = () => useContext(LangContext);
/** true when Chinese lines should be drawn */
export const useZh = () => useContext(LangContext) !== "en";
