\begin{table}
\centering
\begin{tabular}{l|l|l|l}
\hline
                                      & \textbf{n-gram} & \textbf{FER} & \textbf{PER} \\
\hline
Baseline + TIMIT                      & -- & \textbf{10.2\%} & 22.5\%  \\
\hline
No silences                            & 5 & \textbf{10.2\%} & 22.2\%  \\
\hline
\multicolumn{4}{c}{PSST and TIMIT without silence} \\
\hline
CMUdict-end      & 5 & \textbf{10.2\%} & \textbf{22.1\%}  \\
\hline
\multicolumn{4}{c}{Unmodified PSST and TIMIT} \\
\hline
Unmodified CMUdict                     & 5 & 10.3\% & 22.4\%  \\
CMUdict-end    & 5 & \textbf{10.2\%} & 22.2\%  \\
\hline
\end{tabular}
\caption{Results of different language model configurations.}
\label{tab:lmresults}
\end{table}