(* ::Package:: *)

(* ::Section:: *)
(*Clark's moment-matching algorithm*)


(* ::Text:: *)
(*This notebooks illustrates the algorithm proposed in C.E. Clark's paper The Greatest of a Finite Set of Random Variables.*)


(* ::Subsection:: *)
(*The non-degenerate case*)


(* ::Text:: *)
(*First, define the functions as in the original paper (with the exception that we are interested in min, rather than max!). The degenerate case a = 0 will be considered later.*)


(* ::Input:: *)
(*a[\[Sigma]1_, \[Sigma]2_, \[Rho]_]:= Sqrt[(\[Sigma]1-\[Sigma]2)^2+2\[Sigma]1*\[Sigma]2*(1-\[Rho])];*)
(*\[Alpha][\[Mu]1_, \[Mu]2_, \[Sigma]1_, \[Sigma]2_, \[Rho]_] :=(\[Mu]2 - \[Mu]1)/a[\[Sigma]1, \[Sigma]2, \[Rho]];*)
(*\[Phi][x_]:=1/Sqrt[2\[Pi]] Exp[-x^2/2];*)
(*\[CapitalPhi][x_]:=1/2 (1+Erf[x/Sqrt[2]]);*)
(**)
(*\[Nu]1[\[Mu]1_, \[Mu]2_, \[Sigma]1_, \[Sigma]2_, \[Rho]_] := Block[*)
(*{a0 = a[\[Sigma]1, \[Sigma]2, \[Rho]], *)
(*\[Alpha]0 = \[Alpha][\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]]},*)
(**)
(*\[Mu]1 \[CapitalPhi][\[Alpha]0] + \[Mu]2 \[CapitalPhi][-\[Alpha]0] - a0 * \[Phi][\[Alpha]0]*)
(*];*)
(**)
(*\[Nu]2[\[Mu]1_, \[Mu]2_, \[Sigma]1_, \[Sigma]2_, \[Rho]_] := Block[*)
(*{*)
(*a0 = a[\[Sigma]1, \[Sigma]2, \[Rho]], *)
(*\[Alpha]0 = \[Alpha][\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]]*)
(*},*)
(**)
(*(\[Mu]1^2+\[Sigma]1^2)\[CapitalPhi][\[Alpha]0] + (\[Mu]2^2+\[Sigma]2^2)\[CapitalPhi][-\[Alpha]0] - (\[Mu]1 + \[Mu]2) a0 * \[Phi][\[Alpha]0]*)
(*];*)
(**)
(*r[\[Mu]1_, \[Mu]2_, \[Sigma]1_, \[Sigma]2_, \[Rho]_, \[Rho]1_, \[Rho]2_]:=Block[*)
(*{*)
(*\[Alpha]0 = \[Alpha][\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]],*)
(*\[Nu]10=\[Nu]1[\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]],*)
(*\[Nu]20=\[Nu]2[\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]]*)
(*},*)
(**)
(*(\[Sigma]1 * \[Rho]1 *\[CapitalPhi][\[Alpha]0] +\[Sigma]2 * \[Rho]2 * \[CapitalPhi][-\[Alpha]0] )/Sqrt[\[Nu]20-\[Nu]10^2]*)
(*];*)


(* ::Text:: *)
(*Calculations of gradients can be handled automatically now. (These are quite long expressions!)*)


(* ::Input:: *)
(*FullSimplify[Grad[*)
(*\[Nu]1[\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]],*)
(*{\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]}*)
(*]]*)
(**)
(*FullSimplify[Grad[*)
(*\[Nu]2[\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]],*)
(*{\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho]}*)
(*]]*)
(**)
(*Simplify[Grad[*)
(*r[\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho], \[Rho]1, \[Rho]2],*)
(*{\[Mu]1, \[Mu]2, \[Sigma]1, \[Sigma]2, \[Rho], \[Rho]1, \[Rho]2}*)
(*]]*)
(**)


(* ::Subsection:: *)
(*The degenerate case*)


(* ::Text:: *)
(*If (and only if) \[Sigma]1-\[Sigma]2  = \[Rho] - 1  = 0,  we have a = 0 and we cannot use the expressions above. As both variables have the same standard deviation and are perfectly correlated, we have \[Nu]1 = min(\[Mu]1, \[Mu]2]) and \[Nu]2 = \[Nu]1^2+\[Sigma]1^2.*)
(*There are now two important questions:*)
(**)
(*1. What is the correct expression for r in that case?*)
(*2. How do the gradients of \[Nu]1, \[Nu]2, and r look like?*)


(* ::Subsubsection:: *)
(*Expression for r*)


(* ::Text:: *)
(*In this case, \[Rho]1 = \[Rho]2, and this is the correlation we should report.*)


(* ::Subsubsection:: *)
(*Gradients of \[Nu]1 and \[Nu]2*)


(* ::Text:: *)
(*Without the loss of generality, we may assume that \[Mu]1 \[GreaterSlantEqual] \[Mu]2. It's easy in that case to differentiate \[Nu]1=min(\[Mu]1, \[Mu]2) with to respect to \[Mu]1 or to \[Mu]2. (The derivative is 0 and 1, respectively, modulo problems that it's not differentiable at some point. There is also a case that \[Mu]s are equal. Possible TODO).*)
(**)
(*Let's differentiate it with respect to \[Rho]:*)


(* ::Input:: *)
(*Assuming[h > 0 && \[Mu]1 >= \[Mu]2 && \[Sigma] > 0,*)
(*Simplify[*)
(*Series[(\[Mu]2-\[Nu]1[\[Mu]1, \[Mu]2, \[Sigma], \[Sigma], 1-h])/h,{h, 0, 1}]*)
(*]*)
(*]*)


(* ::Text:: *)
(*We see an important thing: if \[Mu]s are different, then this derivative is 0. *)
(**)
(*Let's now calculate this derivative with respect to \[Sigma]1]. Calculation for \[Sigma]2 is similar. *)


(* ::Input:: *)
(*Assuming[h > 0 && \[Mu]1 >=  \[Mu]2 && \[Sigma] > 0,*)
(*Simplify[*)
(*Series[(\[Mu]2-\[Nu]1[\[Mu]1, \[Mu]2, \[Sigma]-h, \[Sigma], 1])/h, *)
(*{h, 0, 1}*)
(*]*)
(*]*)
(*]*)


(* ::Text:: *)
(*Right, again we see the same behaviour: if \[Mu]s are different, then the derivative is 0.*)
(**)
(*Let's do similar calculations for \[Nu]2 now.*)


(* ::Input:: *)
(*Assuming[h > 0 && \[Mu]1 >=  \[Mu]2 && \[Sigma] > 0,*)
(*Simplify[*)
(*Series[( \[Mu]2^2+\[Sigma]^2-\[Nu]2[\[Mu]1, \[Mu]2, \[Sigma], \[Sigma], 1-h])/h, *)
(*{h, 0, 1}*)
(*]*)
(*]*)
(*]*)


(* ::Input:: *)
(*Assuming[h > 0 && \[Mu]1 >=  \[Mu]2 && \[Sigma] > 0,*)
(*Simplify[*)
(*Series[(\[Mu]2^2+\[Sigma]^2-\[Nu]2[\[Mu]1, \[Mu]2, \[Sigma]-h, \[Sigma], 1])/h, *)
(*{h, 0, 1}*)
(*]*)
(*]*)
(*]*)


(* ::Text:: *)
(*Ok, this is interesting -- here the derivative is 2\[Sigma], so it's non-zero. *)


(* ::Subsubsection:: *)
(*Do limits commute with gradients in this case?*)


(* ::Text:: *)
(*It happens from time to time that gradients and limits commute (as in Moore-Osgood theorem or Schwarz's theorem). Let's take our analytic expression (that don't hold for the degenerate case) and pass to the degenerate limit with them.*)
(*Caution: the degenerate limit is (\[Sigma]1, \[Sigma]2, \[Rho]) -> (\[Sigma], \[Sigma],1). It's not the same as an iterative calculation of the limits! (But we'll do it anyway, as it's simpler). *)
(**)
(*TODO*)


(* ::Input:: *)
(**)
