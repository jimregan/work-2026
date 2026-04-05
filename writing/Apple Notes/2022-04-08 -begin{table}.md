\begin{table}
\centering
\begin{tabular}{l|l|l|l}
\hline
                                      & \textbf{n-gram} & \textbf{FER} & \textbf{PER} \\
\hline
Baseline + TIMIT                      & -- & \textbf{10.2\%} & 22.5\%  \\
\hline
%\multirow{3}{*}{All silences}         & 4 & 10.5\% & 23.0\%  \\
%                                      & 5 & 10.5\% & 22.6\%  \\
%                                      & 6 & 10.3\% & 22.3\%  \\
%\hline
\multirow{3}{*}{No silences}          & 4 & 10.3\% & 22.6\%  \\
                                      & 5 & \textbf{10.2\%} & 22.2\%  \\
                                      & 6 & \textbf{10.2\%} & 22.4\%  \\
\hline
\multicolumn{4}{c}{PSST and TIMIT without silence} \\
\hline
\multirow{3}{*}{CMUdict-end}   & 4 & 10.3\% & 22.6\%  \\
                                      & 5 & \textbf{10.2\%} & \textbf{22.1\%}  \\
                                      & 6 & \textbf{10.2\%} & 22.3\%  \\
%\multirow{3}{*}{CMUdict-start}   & 4 & 10.4\% & 22.6\%  \\
%                                      & 5 & 10.3\% & 22.4\%  \\
%                                      & 6 & 10.3\% & 22.3\%  \\
%\multirow{3}{*}{CMUdict-both} & 4 & 10.4\% & 22.7\%  \\
%                                      & 5 & 10.4\% & 22.3\%  \\
%                                      & 6 & 10.3\% & 22.3\%  \\
\hline
\multicolumn{4}{c}{Unmodified PSST and TIMIT} \\
\hline
\multirow{3}{*}{Unmodified CMUdict}   & 4 & 10.3\% & 22.8\%  \\
                                      & 5 & 10.3\% & 22.4\%  \\
                                      & 6 & \textbf{10.2\%} & 22.4\%  \\
\multirow{3}{*}{CMUdict-end}    & 4 & 10.3\% & 22.7\%  \\
                                      & 5 & \textbf{10.2\%} & 22.2\%  \\
                                      & 6 & \textbf{10.2\%} & 22.3\%  \\
%\multirow{3}{*}{CMUdict-start}    & 4 & 10.5\% & 22.8\%  \\
%                                      & 5 & 10.4\% & 22.5\%  \\
%                                      & 6 & 10.3\% & 22.4\%  \\
%\multirow{3}{*}{CMUdict-both}    & 4 & 10.5\% & 22.8\%  \\
%                                      & 5 & 10.4\% & 22.4\%  \\
%                                      & 6 & 10.4\% & 22.4\%  \\
\hline
\end{tabular}
\caption{Results of different language model configurations.}
\label{tab:lmresults}
\end{table}