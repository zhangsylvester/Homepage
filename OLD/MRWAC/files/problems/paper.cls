\ProvidesClass{paper}

\newif\ifpaper@beamer  \paper@beamerfalse
\DeclareOption{beamer}{\paper@beamertrue \PassOptionsToPackage{beamer}{notation}}
\ProcessOptions\relax

\ifpaper@beamer
 \LoadClass[xcolor={dvipsnames}]{beamer}
 \RequirePackage[backend=bibtex,style=authoryear,maxnames=4,maxalphanames=4,backref=true]{biblatex}
\else
 \LoadClass[12pt,reqno]{amsart}
 \RequirePackage[letterpaper,margin=1in]{geometry}
 \RequirePackage[alphabetic,backrefs]{amsrefs}
% \RequirePackage[backend=bibtex,backref=true,style=alphabetic,maxnames=4,maxalphanames=4]{biblatex}
 \RequirePackage[colorlinks=true,hyperindex,linkcolor=magenta,pagebackref=true,citecolor=cyan]{hyperref}
\fi

\RequirePackage[utf8]{inputenc}
\RequirePackage[english]{babel}
%\RequirePackage{stix} % Times New Roman

\RequirePackage{graphicx}
\RequirePackage{microtype}
\RequirePackage{fancyhdr}
%\RequirePackage{titlesec} % allergen
\RequirePackage{wrapfig}
\RequirePackage{color}

\RequirePackage{notation}

%\addbibresource{sources.bib}
%\renewbibmacro{in:}{}

%\renewcommand{\bibfont}{\footnotesize} % \scriptsize
%\setlength\bibitemsep{0pt}
%\setlength\bibparsep{0pt}

\reversemarginpar
\setlength{\marginparwidth}{0.75in}
%\renewcommand{\marginpar}[2][]{} % Skip margin notes

