%% missingbook.cls -- "The Missing Textbooks" series class
%% A KOMA-Script based textbook design with a margin column for side notes,
%% boxed theorems, "The Move" boxes, a dissolution-test table, exercises,
%% per-chapter bibliographies (biblatex/biber) and a rising-sea chapter motif.
%%
%% Engine: pdfLaTeX.  Bibliography: biber.  Index: makeindex (run by imakeidx).
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{missingbook}[2026/10/17 v1.0 The Missing Textbooks series class]

%------------------------------------------------------------------------------
% Options
%------------------------------------------------------------------------------
\newcommand*{\mb@font}{palatino}
\DeclareOption{palatino}{\renewcommand*{\mb@font}{palatino}}
\DeclareOption{cochineal}{\renewcommand*{\mb@font}{cochineal}}
\DeclareOption{garamond}{\renewcommand*{\mb@font}{garamond}}
\DeclareOption{charter}{\renewcommand*{\mb@font}{charter}}
\DeclareOption{libertinus}{\renewcommand*{\mb@font}{libertinus}}
\newif\ifmb@draft\mb@draftfalse
\DeclareOption{draft}{\mb@drafttrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ProcessOptions\relax

\LoadClass[fontsize=10.5pt,twoside=false,open=any,numbers=noenddot,
  headings=optiontoheadandtoc,chapterprefix=false,index=totoc]{scrbook}

%------------------------------------------------------------------------------
% Page layout: text block plus a margin column for side notes
%------------------------------------------------------------------------------
\RequirePackage[paperwidth=210mm,paperheight=280mm,
  left=24mm,textwidth=120mm,marginparsep=7mm,marginparwidth=44mm,
  top=25mm,bottom=31mm,headheight=15pt,headsep=8mm,footskip=12mm]{geometry}
\newcommand*{\mb@paperwidthmm}{210}

%------------------------------------------------------------------------------
% Fonts
%------------------------------------------------------------------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[english]{babel}
\RequirePackage{amsmath}
\RequirePackage{amsthm}
\def\mb@tmp{palatino}\ifx\mb@font\mb@tmp
  \RequirePackage{newpxtext}
  \RequirePackage[vvarbb,bigdelims]{newpxmath}
\fi
\def\mb@tmp{cochineal}\ifx\mb@font\mb@tmp
  \RequirePackage[p]{cochineal}
  \RequirePackage[cochineal,vvarbb,bigdelims]{newtxmath}
\fi
\def\mb@tmp{garamond}\ifx\mb@font\mb@tmp
  \RequirePackage[lining]{ebgaramond}
  \RequirePackage[ebgaramond,vvarbb,bigdelims]{newtxmath}
\fi
\def\mb@tmp{charter}\ifx\mb@font\mb@tmp
  \RequirePackage[lining]{XCharter}
  \RequirePackage[xcharter,vvarbb,bigdelims]{newtxmath}
\fi
\def\mb@tmp{libertinus}\ifx\mb@font\mb@tmp
  \RequirePackage{libertinus}
  \RequirePackage[vvarbb]{libertinust1math}
\fi
\RequirePackage[semibold,scale=0.92]{sourcesans}
\RequirePackage{mathtools}
\RequirePackage{bm}
\RequirePackage[final,protrusion=true,expansion=true,tracking=smallcaps]{microtype}
\SetTracking{encoding=*,shape=sc}{45}
\linespread{1.06}

%------------------------------------------------------------------------------
% Colours
%------------------------------------------------------------------------------
\RequirePackage{xcolor}
\definecolor{sea}{RGB}{0,92,122}
\definecolor{seadeep}{RGB}{0,58,80}
\definecolor{seafoam}{RGB}{234,244,247}
\definecolor{seamist}{RGB}{214,234,240}
\definecolor{sand}{RGB}{249,245,236}
\definecolor{sandline}{RGB}{205,186,142}
\definecolor{ink}{RGB}{34,34,34}
\definecolor{muted}{RGB}{108,108,108}
\color{ink}

%------------------------------------------------------------------------------
% General packages
%------------------------------------------------------------------------------
\RequirePackage{etoolbox}
\RequirePackage{enumitem}
\RequirePackage{booktabs,array,tabularx,multirow}
\RequirePackage{graphicx}
\RequirePackage{changepage}
\RequirePackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning,decorations.pathreplacing,decorations.pathmorphing,
  patterns,shapes.geometric,matrix,fit,backgrounds,intersections}
\RequirePackage[most]{tcolorbox}
\RequirePackage{caption}
\captionsetup{font=small,labelfont={color=sea,bf},labelsep=period,format=plain}
\RequirePackage{csquotes}
\RequirePackage[backend=biber,style=authoryear-comp,maxcitenames=2,maxbibnames=8,
  uniquename=false,uniquelist=false,giveninits=true,dashed=false,doi=false,url=false,
  isbn=false,eprint=false,date=year]{biblatex}
\RequirePackage{imakeidx}
\makeindex[intoc,columns=2,title=Index]
\setlist{itemsep=0.25ex,topsep=0.6ex}
\setlist[enumerate]{label=(\roman*),leftmargin=*}
\setlist[itemize]{label={\color{sea}\textbullet},leftmargin=*}

%------------------------------------------------------------------------------
% Headings
%------------------------------------------------------------------------------
\setkomafont{disposition}{\normalfont\color{ink}}
\setkomafont{chapter}{\normalfont\fontsize{26}{31}\selectfont}
\setkomafont{section}{\normalfont\Large\bfseries}
\setkomafont{subsection}{\normalfont\large\bfseries}
\setkomafont{subsubsection}{\normalfont\normalsize\bfseries\itshape}
\setkomafont{paragraph}{\normalfont\normalsize\scshape\color{sea}}
\setkomafont{part}{\normalfont\color{seadeep}\fontsize{30}{36}\selectfont}
\setkomafont{partnumber}{\normalfont\Large\scshape\color{sea}}
\setkomafont{chapterentry}{\normalfont\color{ink}}
\setkomafont{chapterentrypagenumber}{\normalfont\color{sea}}
\setkomafont{partentry}{\normalfont\scshape\color{sea}}
\setkomafont{partentrypagenumber}{\normalfont\color{sea}}
\AtEndPreamble{\DeclareTOCStyleEntries[linefill=\hfill]{tocline}{section,subsection}}
\setcounter{tocdepth}{1}

\RedeclareSectionCommand[beforeskip=0pt,afterskip=0.6\baselineskip]{chapter}
\RedeclareSectionCommand[beforeskip=-2.4ex plus -0.8ex minus -0.3ex,afterskip=1.1ex plus 0.2ex]{section}
\RedeclareSectionCommand[beforeskip=-2ex plus -0.6ex minus -0.2ex,afterskip=0.9ex plus 0.2ex]{subsection}
\RedeclareSectionCommand[beforeskip=-1.6ex plus -0.5ex minus -0.2ex,afterskip=0.6ex plus 0.1ex]{subsubsection}
\RedeclareSectionCommand[beforeskip=1.2ex plus 0.4ex minus 0.2ex,afterskip=-0.8em]{paragraph}

% keep headings with at least a few lines of what follows
\RequirePackage{needspace}
\AddtoDoHook{heading/preinit/section}{\Needspace{6\baselineskip}\@gobble}
\AddtoDoHook{heading/preinit/subsection}{\Needspace{5\baselineskip}\@gobble}
% section numbers hang in the left margin
\renewcommand*{\sectionformat}{\makebox[0pt][r]{\textcolor{sea}{\thesection}\hspace*{0.9em}}}
\renewcommand*{\subsectionformat}{\makebox[0pt][r]{\textcolor{sea}{\thesubsection}\hspace*{0.9em}}}
\renewcommand*{\chapterformat}{\thechapter}
\newcommand*{\mb@chapnumfont}{\normalfont\fontsize{64}{64}\selectfont\color{sea!42}}
\renewcommand*{\chapterlinesformat}[3]{%
  \Ifstr{#1}{chapter}{%
    \makebox[\linewidth][l]{%
      \parbox[b]{\linewidth}{\raggedright #3}%
      \Ifstr{#2}{}{}{\hspace*{\marginparsep}\parbox[b]{\marginparwidth}{\raggedleft\mb@chapnumfont #2}}%
    }%
  }{\@hangfrom{#2}{#3}}%
}
\renewcommand*{\partformat}{\partname~\thepart}
\renewcommand*{\raggedpart}{\centering}

% thin wave rule across text block and margin column
\newcommand*{\mb@waverule}[1][sea!55]{%
  \par\noindent\makebox[\linewidth][l]{\begin{tikzpicture}
    \draw[#1,line width=0.55pt,decorate,decoration={snake,amplitude=0.8pt,segment length=8pt}]
      (0,0)--(\linewidth+\marginparsep+\marginparwidth,0);
  \end{tikzpicture}}\par}

% the rising sea: a wave band at the bottom of an opening page; #1 = height in mm
\newcommand*{\risingsea}[1]{%
  \begin{tikzpicture}[remember picture,overlay]
    \begin{scope}[shift={(current page.south west)}]
      \fill[seafoam] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=160,smooth]
        (\x/10,{(#1+1.4*sin(\x*4.6))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
      \fill[seamist] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=160,smooth]
        (\x/10,{(#1-3.2+1.1*sin(\x*6.5+70))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
      \draw[sea!35,line width=0.4pt] plot[domain=0:\mb@paperwidthmm,samples=160,smooth]
        (\x/10,{(#1+1.4*sin(\x*4.6))/10});
    \end{scope}
  \end{tikzpicture}}

%------------------------------------------------------------------------------
% Case chapters
%   \casechapter{Title}{Subtitle}{Move}{label}
%   \specialchapter{Title}{Subtitle}{tide height in mm}   (unnumbered)
%------------------------------------------------------------------------------
\newcommand*{\mbtocsubtitle}[1]{%
  \par\nopagebreak\vspace{-0.15ex}{\leftskip=2.3em\small\itshape\color{muted}#1\par}\vspace{0.35ex}}
\newcommand{\mb@subtitleblock}[2]{%
  \vspace*{-0.2\baselineskip}%
  {\raggedright\large\itshape\color{muted}#1\par}%
  \Ifstr{#2}{}{}{%
    \vspace{0.55\baselineskip}%
    {\raggedright\small\textsc{\color{sea}the move}\enspace{\color{sea}\rule[0.32ex]{1.6em}{0.4pt}}\enspace\textsc{#2}\par}}%
  \vspace{0.5\baselineskip}%
  \mb@waverule
  \vspace{0.9\baselineskip}}
\newcommand{\casechapter}[4]{%
  \chapter[head={#1},tocentry={#1}]{#1}\label{#4}%
  \addtocontents{toc}{\protect\mbtocsubtitle{#2}}%
  \risingsea{\fpeval{6+1.25*\value{chapter}}}%
  \mb@subtitleblock{#2}{#3}%
  \@afterindentfalse\@afterheading}
\newcommand{\specialchapter}[3]{%
  \addchap[head={#1},tocentry={#1}]{#1}%
  \addtocontents{toc}{\protect\mbtocsubtitle{#2}}%
  \risingsea{#3}%
  \mb@subtitleblock{#2}{}%
  \@afterindentfalse\@afterheading}

% part pages: \casepart{Title}{blurb}{tide height in mm}
\newcommand{\casepart}[3]{%
  \setpartpreamble[c][0.78\textwidth]{%
    \vspace{2.2\baselineskip}{\centering\mb@waverule[sea!40]\par}\vspace{1.2\baselineskip}%
    {\normalfont\itshape\color{muted}\setlength{\parindent}{0pt}#2\par}%
    \risingsea{#3}}%
  \part{#1}}

%------------------------------------------------------------------------------
% Running heads
%------------------------------------------------------------------------------
\RequirePackage[headsepline=0.3pt]{scrlayer-scrpage}
\KOMAoptions{headwidth=textwithmarginpar}
\setkomafont{pageheadfoot}{\normalfont\footnotesize\color{muted}}
\setkomafont{pagenumber}{\normalfont\footnotesize\bfseries\color{sea}}
\setkomafont{headsepline}{\color{sea!25}}
\clearpairofpagestyles
\automark[chapter]{chapter}
\renewcommand*{\chaptermarkformat}{\thechapter\enskip\textperiodcentered\enskip}
\ihead{\scshape\MakeLowercase{\leftmark}}
\ohead*{\pagemark}
\renewcommand*{\chapterpagestyle}{plain.scrheadings}
\renewcommand*{\partpagestyle}{empty}

%------------------------------------------------------------------------------
% Side notes in the margin column
%------------------------------------------------------------------------------
\newcounter{sidenote}
\counterwithin*{sidenote}{chapter}
\newcommand*{\mb@marginstyle}{\normalfont\footnotesize\linespread{1.0}\selectfont
  \color{ink!88}\raggedright\setlength{\parindent}{0pt}\hbadness=10000}
\newcommand{\sidenote}[1]{%
  \refstepcounter{sidenote}%
  \textsuperscript{\textcolor{sea}{\sffamily\thesidenote}}%
  \marginpar{\mb@marginstyle\makebox[0pt][r]{\textsuperscript{\textcolor{sea}{\sffamily\thesidenote}}\,}#1}}
\newcommand{\marginaside}[1]{\marginpar{\mb@marginstyle #1}}
\setlength{\marginparpush}{7pt}

% wide material spanning text block and margin column
\newenvironment{fullwidth}{\begin{adjustwidth}{0pt}{-\dimexpr\marginparsep+\marginparwidth\relax}}{\end{adjustwidth}}

%------------------------------------------------------------------------------
% Theorems
%------------------------------------------------------------------------------
\newtheoremstyle{mbplain}{0pt}{0pt}{\itshape}{}{\bfseries}{.}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{\normalfont\ ({\let\textcite\cite #3})}}
\newtheoremstyle{mbdefinition}{0pt}{0pt}{\normalfont}{}{\bfseries}{.}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{\normalfont\ ({\let\textcite\cite #3})}}
\newtheoremstyle{mbremark}{0.8ex plus 0.3ex}{0.8ex plus 0.3ex}{\normalfont}{}{\scshape}{.}{0.5em}%
  {\thmname{#1}\thmnumber{ #2}\thmnote{\normalfont\ ({\let\textcite\cite #3})}}

\RequirePackage[hidelinks]{hyperref}
\hypersetup{colorlinks=true,linkcolor=sea,citecolor=sea,urlcolor=sea,
  bookmarksnumbered=true,pdfcreator={missingbook.cls}}
\RequirePackage{bookmark}
\RequirePackage[nameinlink,capitalise,noabbrev]{cleveref}

% Theorem-like environments share one counter. Each gets an alias counter
% (aliascnt), so that \cref prints "Lemma 3.2" rather than "Theorem 3.2".
\RequirePackage{aliascnt}
\newcommand*{\mb@newsharedtheorem}[2]{%
  \newaliascnt{#1}{theorem}\newtheorem{#1}[#1]{#2}\aliascntresetthe{#1}}
\theoremstyle{mbplain}
\newtheorem{theorem}{Theorem}[chapter]
\mb@newsharedtheorem{lemma}{Lemma}
\mb@newsharedtheorem{proposition}{Proposition}
\mb@newsharedtheorem{corollary}{Corollary}
\mb@newsharedtheorem{conjecture}{Conjecture}
\mb@newsharedtheorem{claim}{Claim}
\theoremstyle{mbdefinition}
\mb@newsharedtheorem{definition}{Definition}
\mb@newsharedtheorem{problem}{Problem}
\theoremstyle{mbremark}
\mb@newsharedtheorem{example}{Example}
\mb@newsharedtheorem{remark}{Remark}
\theoremstyle{mbplain}
\newtheorem*{theorem*}{Theorem}
\newtheorem*{lemma*}{Lemma}
\newtheorem*{proposition*}{Proposition}
\newtheorem*{corollary*}{Corollary}
\newtheorem*{conjecture*}{Conjecture}
\theoremstyle{mbdefinition}
\newtheorem*{definition*}{Definition}
\newtheorem*{problem*}{Problem}
\theoremstyle{mbremark}
\newtheorem*{example*}{Example}
\newtheorem*{remark*}{Remark}

\crefname{theorem}{Theorem}{Theorems}
\crefname{lemma}{Lemma}{Lemmas}
\crefname{proposition}{Proposition}{Propositions}
\crefname{corollary}{Corollary}{Corollaries}
\crefname{conjecture}{Conjecture}{Conjectures}
\crefname{claim}{Claim}{Claims}
\crefname{definition}{Definition}{Definitions}
\crefname{problem}{Problem}{Problems}
\crefname{example}{Example}{Examples}
\crefname{remark}{Remark}{Remarks}
\crefname{chapter}{Chapter}{Chapters}
\crefname{section}{Section}{Sections}
\crefname{subsection}{Section}{Sections}
\crefname{figure}{Figure}{Figures}
\crefname{table}{Table}{Tables}
\crefname{equation}{}{}
\crefformat{equation}{#2(#1)#3}
\crefrangeformat{equation}{#3(#1)#4--#5(#2)#6}

\tcbset{mbthm/.style={enhanced,breakable,colback=seafoam,frame hidden,boxrule=0pt,arc=0pt,
    borderline west={1.3pt}{0pt}{sea},left=3.2mm,right=3mm,top=1.6mm,bottom=1.6mm,
    before skip=1.3ex plus 0.5ex,after skip=1.3ex plus 0.5ex},
  mbdef/.style={enhanced,breakable,colback=sand,frame hidden,boxrule=0pt,arc=0pt,
    borderline west={1.3pt}{0pt}{sandline},left=3.2mm,right=3mm,top=1.6mm,bottom=1.6mm,
    before skip=1.3ex plus 0.5ex,after skip=1.3ex plus 0.5ex}}
\tcolorboxenvironment{theorem}{mbthm}
\tcolorboxenvironment{lemma}{mbthm}
\tcolorboxenvironment{proposition}{mbthm}
\tcolorboxenvironment{corollary}{mbthm}
\tcolorboxenvironment{conjecture}{mbthm}
\tcolorboxenvironment{claim}{mbthm}
\tcolorboxenvironment{theorem*}{mbthm}
\tcolorboxenvironment{lemma*}{mbthm}
\tcolorboxenvironment{proposition*}{mbthm}
\tcolorboxenvironment{corollary*}{mbthm}
\tcolorboxenvironment{conjecture*}{mbthm}
\tcolorboxenvironment{definition}{mbdef}
\tcolorboxenvironment{definition*}{mbdef}
\tcolorboxenvironment{problem}{mbdef}
\tcolorboxenvironment{problem*}{mbdef}

% proofs: small-caps head, Halmos tombstone
\renewcommand{\qedsymbol}{\ensuremath{\blacksquare}}
\renewenvironment{proof}[1][Proof]{\par\pushQED{\qed}\normalfont\topsep5\p@\@plus5\p@\relax
  \trivlist\item[\hskip\labelsep\scshape #1.]\ignorespaces}
  {\popQED\endtrivlist\@endpefalse}
\newenvironment{proofsketch}[1][Proof sketch]{\begin{proof}[#1]}{\end{proof}}

%------------------------------------------------------------------------------
% Signature boxes
%------------------------------------------------------------------------------
\newtcolorbox{chaptersummary}{enhanced,breakable,colback=sand,frame hidden,boxrule=0pt,arc=0pt,
  left=4mm,right=4mm,top=2.6mm,bottom=2.6mm,fontupper=\normalfont,
  before upper={\textsc{\color{sea}in this chapter.}\enspace},before skip=0pt,after skip=1.6ex}

\newtcolorbox{move}[1]{enhanced,breakable,colback=seafoam,colframe=sea,boxrule=0pt,
  toprule=0pt,bottomrule=0pt,leftrule=0pt,rightrule=0pt,arc=0pt,
  left=4mm,right=4mm,top=2.4mm,bottom=2.8mm,
  title={\textsc{the move}\enspace\textperiodcentered\enspace\textsc{#1}},
  fonttitle=\normalfont\large,coltitle=white,colbacktitle=sea,
  toptitle=1.2mm,bottomtitle=1.2mm,before skip=1.6ex,after skip=1.6ex}
\newcommand{\movepart}[1]{\par\smallskip\noindent{\scshape\color{sea}#1.}\enspace}

\newtcolorbox{aside}[1][]{enhanced,breakable,colback=black!3,frame hidden,boxrule=0pt,arc=0pt,
  borderline west={0.6pt}{0pt}{muted},left=3.2mm,right=3mm,top=1.6mm,bottom=1.6mm,
  fontupper=\small,before skip=1.2ex,after skip=1.2ex,
  before upper={\Ifstr{#1}{}{}{\textsc{\color{muted}#1.}\enspace}}}

% \dissolutiontest{shorter proofs}{fewer assumptions}{unification}{new reach}
\newcommand*{\mb@testlabel}[1]{\textsc{\textcolor{sea}{#1}}}
\newcommand{\mb@testrow}[2]{\par\noindent\hangindent=10.8em\hangafter=1
  \makebox[10.8em][l]{\mb@testlabel{#1}}#2\par\vspace{0.45ex}}
\newcommand{\dissolutiontest}[4]{%
  \begin{tcolorbox}[enhanced,breakable,colback=white,colframe=sea!45,boxrule=0.5pt,arc=0pt,
    left=3mm,right=3mm,top=1.6mm,bottom=1.2mm,title={\textsc{dissolution test}},
    fonttitle=\normalfont,coltitle=sea,colbacktitle=seafoam,toptitle=0.8mm,bottomtitle=0.8mm,
    before skip=1.6ex,after skip=1.6ex,fontupper=\small]
  \mb@testrow{shorter proofs}{#1}%
  \mb@testrow{fewer assumptions}{#2}%
  \mb@testrow{unification}{#3}%
  \mb@testrow{new reach}{#4}%
  \end{tcolorbox}}

%------------------------------------------------------------------------------
% Exercises, notes, references
%------------------------------------------------------------------------------
\newlist{mbexerciselist}{enumerate}{1}
\setlist[mbexerciselist]{label={\textcolor{sea}{\bfseries\thechapter.\arabic*}},ref={\thechapter.\arabic*},
    leftmargin=*,widest=99,itemsep=0.9ex,topsep=0.8ex}
\crefname{mbexerciselisti}{Exercise}{Exercises}
\newenvironment{exercises}{%
  \section*{Exercises}\addcontentsline{toc}{section}{Exercises}%
  \begin{mbexerciselist}}
  {\end{mbexerciselist}}
\newcommand{\consolidation}{{\scshape\color{sea}consolidation.}\enspace}
\newcommand{\transfer}{{\scshape\color{sea}transfer.}\enspace}
\newcommand{\research}{{\scshape\color{sea}research.}\enspace}
\newcommand{\hard}{{\color{sea}\ensuremath{\star}}\enspace}
\newcommand{\harder}{{\color{sea}\ensuremath{\star\star}}\enspace}
\newcommand{\hint}[1]{{\itshape (Hint: #1)}}

\newcommand{\notesandrefs}{\section*{Notes and References}\addcontentsline{toc}{section}{Notes and References}}
\defbibheading{mbchapter}[References]{\subsection*{#1}}
\newcommand{\printchapterbib}{\printbibliography[heading=mbchapter]}
\renewcommand*{\bibfont}{\small}
\setlength{\bibitemsep}{0.35ex}

% first use of a technical term: \term{uniform convergence}
\newcommand{\term}[1]{\emph{#1}}

%------------------------------------------------------------------------------
% Mathematics
%------------------------------------------------------------------------------
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\F}{\mathbb{F}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\eps}{\varepsilon}
\newcommand{\ind}{\mathbf{1}}
\newcommand{\dd}{\,\mathrm{d}}
\newcommand{\defeq}{\coloneqq}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator{\rank}{rank}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\sgn}{sgn}
\DeclareMathOperator{\supp}{supp}
\DeclareMathOperator{\poly}{poly}
\DeclareMathOperator{\conv}{conv}
\DeclareMathOperator{\diam}{diam}
\DeclareMathOperator{\dist}{dist}
\DeclarePairedDelimiter{\abs}{\lvert}{\rvert}
\DeclarePairedDelimiter{\norm}{\lVert}{\rVert}
% \ip{a,b} and \ip{a}{b} both give <a, b>; \ip*{..} auto-sizes, \ip[\big]{..} fixes the size.
\NewDocumentCommand{\ip}{s o m g}{%
  \IfBooleanTF{#1}%
    {\left\langle #3\IfValueT{#4}{, #4}\right\rangle}%
    {\IfNoValueTF{#2}%
      {\langle #3\IfValueT{#4}{, #4}\rangle}%
      {\mathopen{#2\langle}#3\IfValueT{#4}{, #4}\mathclose{#2\rangle}}}}
\DeclarePairedDelimiter{\ceil}{\lceil}{\rceil}
\DeclarePairedDelimiter{\floor}{\lfloor}{\rfloor}
\newcommand{\cA}{\mathcal{A}}\newcommand{\cB}{\mathcal{B}}\newcommand{\cC}{\mathcal{C}}
\newcommand{\cD}{\mathcal{D}}\newcommand{\cE}{\mathcal{E}}\newcommand{\cF}{\mathcal{F}}
\newcommand{\cG}{\mathcal{G}}\newcommand{\cH}{\mathcal{H}}\newcommand{\cI}{\mathcal{I}}
\newcommand{\cL}{\mathcal{L}}\newcommand{\cM}{\mathcal{M}}\newcommand{\cN}{\mathcal{N}}
\newcommand{\cP}{\mathcal{P}}\newcommand{\cS}{\mathcal{S}}\newcommand{\cT}{\mathcal{T}}
\newcommand{\cU}{\mathcal{U}}\newcommand{\cW}{\mathcal{W}}\newcommand{\cX}{\mathcal{X}}
\newcommand{\cY}{\mathcal{Y}}\newcommand{\cZ}{\mathcal{Z}}

%------------------------------------------------------------------------------
% Front matter helpers
%------------------------------------------------------------------------------
\newcommand{\mbtableofcontents}{{\hypersetup{linkcolor=ink}\tableofcontents}}

% \mbtitlepage{Title}{Subtitle}{Author}{Imprint line}
\newcommand{\mbtitlepage}[4]{%
  \begin{titlepage}
  \begin{tikzpicture}[remember picture,overlay]
    \begin{scope}[shift={(current page.south west)}]
      \fill[seafoam] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=200,smooth]
        (\x/10,{(84+2.2*sin(\x*3.1))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
      % the nut, soaking
      \fill[brown!55!black] (15.2,7.6) ellipse (1.25 and 1.55);
      \draw[brown!30!black,line width=0.5pt] (15.2,6.1) .. controls (15.05,7.3) and (15.35,8.0) .. (15.2,9.15);
      \draw[brown!75!black,line width=0.35pt] (14.55,6.6) .. controls (14.35,7.5) and (14.5,8.4) .. (14.8,8.95);
      \draw[brown!75!black,line width=0.35pt] (15.85,6.6) .. controls (16.05,7.5) and (15.9,8.4) .. (15.6,8.95);
      \fill[seamist,opacity=0.92] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=200,smooth]
        (\x/10,{(76+2.0*sin(\x*4.3+40))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
      \draw[sea!45,line width=0.5pt] plot[domain=0:\mb@paperwidthmm,samples=200,smooth]
        (\x/10,{(76+2.0*sin(\x*4.3+40))/10});
      \fill[sea!22,opacity=0.9] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=200,smooth]
        (\x/10,{(52+1.8*sin(\x*5.7+120))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
      \fill[sea!40,opacity=0.9] (0,0) -- plot[domain=0:\mb@paperwidthmm,samples=200,smooth]
        (\x/10,{(28+1.5*sin(\x*7.1+200))/10}) -- (\mb@paperwidthmm/10,0) -- cycle;
    \end{scope}
  \end{tikzpicture}%
  \setlength{\parindent}{0pt}%
  \vspace*{18mm}
  {\large\scshape\color{sea}the missing textbooks\par}
  \vspace{14mm}
  \makebox[\textwidth][l]{\parbox{\dimexpr\textwidth+\marginparsep+\marginparwidth\relax}{\raggedright\hyphenpenalty=10000\exhyphenpenalty=10000
    \fontsize{44}{50}\selectfont\color{ink}#1}}\par
  \vspace{7mm}
  {\LARGE\itshape\color{muted}#2\par}
  \vspace{20mm}
  {\Large #3\par}
  \vfill
  {\small\color{seadeep}#4\par}
  \vspace*{4mm}
  \end{titlepage}}

\endinput
