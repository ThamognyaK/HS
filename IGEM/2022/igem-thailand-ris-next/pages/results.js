function Results({ children }) {
    return (
        <div>
            <div class="titleScreen1">
                <h1 class="titlescreentext">RESULTS</h1>
            </div>
            <div class="body1">
                {/* HEADER */}
                <div class="myheader">
                    <div class="rounded-container">
                        <h1 class="header2">WET LAB</h1>
                    </div>
                    <div class="circle1"></div>
                </div>
                {/* HEADER */}

                {/* PARAGRAPH */}
                <h2 class="header1">Bombolitin Vector Design</h2>
                <p class="indented1">
                    We sub-cloned the bombolitin mRNA sequence from the venom of
                    the bumblebee
                    <i> Bombus ignitus</i> to pGEX-61 by process of DNA
                    synthesis. From this, we found out that bombolitin has a
                    2kDa mature peptide with approximately 18-amino acid
                    residues. Since we were concerned with the stability and
                    activity of the peptide, we decided to modify the peptide by
                    adding 3 amino acids which were phenylalanine, leucine, and
                    tryptophan at the C-terminal to test the stability and
                    activity. We then had to transfer these inserts separately
                    to PET32a for protein expression.
                </p>
                {/* PARAGRAPH */}

                <div class="sizedbox10" />

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-11-at-00-02-45.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}
                <div class="sizedbox60" />

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/wetlabvector2.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}
                <div class="sizedbox60" />

                {/* PARAGRAPH */}
                <h2 class="header1">DNA Amplification of Bombolitin</h2>
                {/* PARAGRAPH */}

                <div class="sizedbox60" />

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/wetlabamplification.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}

                {/* PARAGRAPH */}
                <p class="indented1">
                    We successfully amplified Bombolitin using PCR and checked
                    the results that included the primer using electrophoresis.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />

                {/* PARAGRAPH */}
                <h2 class="header1">DNA Purification of Bombolitin</h2>
                {/* PARAGRAPH */}
                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/wetlabpurificatino.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}
                {/* PARAGRAPH */}
                <p class="indented1">
                    We used the process of PCR purification to achieve the
                    purified DNA of bombolitin. Above is the result of the gel
                    electrophoresis that we performed to check the results of
                    the purified DNA.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />

                {/* PARAGRAPH */}
                <h2 class="header1">Ligation of Bombolitin with pET23-a</h2>
                {/* PARAGRAPH */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/ligation-1.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/ligation-2.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}

                {/* PARAGRAPH */}
                <p class="indented1">
                    From ligating the bombolitin with pET32a, we successfully
                    modified the c-terminal with FFF, LLL, and WWW respectively.
                    After the process of ligation and cut by BamHI and SalI, the
                    results showed that pET32a-Bom LLL succeeded for insertion
                    to pET32a.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />

                {/* PARAGRAPH */}
                <h2 class="header1">Protein Expression of pET32a-Bombolitin</h2>
                {/* PARAGRAPH */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/bombolitinexpress.png"
                        class="fitimage1"
                    />
                </div>
                {/* IMAGE */}

                {/* PARAGRAPH */}
                <p class="indented1">
                    From the protein expression of pET32a-Bombolitin, the
                    results showed that the rBombolitin was expressed after
                    induced by 0.5 mM IPTG at 37 degrees celsius for 4 hours.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />

                {/* PARAGRAPH */}
                <h2 class="header1">
                    Protein Characterization of pET32a-Bombolitin{' '}
                </h2>
                <p class="indented1">
                    Through the process of protein characterization, the results
                    showed that the rBombolitin 22.04 kDa bound to the Ni-NTA
                    column was eluded by 300mM imidazole. We also tested using
                    agar diffusion assay and the result showed that 500 µg of
                    the modified Bombolitin peptide displayed significant
                    activity against
                    <i> Ralstonia solanacearum</i>.
                </p>

                {/* PARAGRAPH */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/proteincharacterization.png"
                        class="fitimage1"
                    />
                </div>

                {/* IMAGE */}

                {/* PARAGRAPH */}
                <p class="indented1">
                    The negative control was buffered saline, which shows no
                    antimicrobial activity. 500ug of original Bombolitin also
                    shows inhibition of <i>Ralstonia</i>, but not as significant
                    as the modified Bombolitin or positive control. For the
                    modified Bombolitin, the diameter of inhibition is 5mm. The
                    positive control used ampicillin which has an inhibition of
                    10mm. We have concluded that the minimum inhibitory
                    concentration of modified Bombolitin is 500ug.
                </p>

                {/* PARAGRAPH */}

                <div class="sizedbox60" />

                {/* HEADER */}
                <div class="myheader">
                    <div class="rounded-container">
                        <h1 class="header2">DRY LAB</h1>
                    </div>
                    <div class="circle1"></div>
                </div>
                {/* HEADER */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/infectintoecoli1.png"
                        class="fitimage1"
                    />
                    <p class="fitimage1">
                        Infect into <i>E.coli</i> TG1 (First round)
                    </p>
                </div>

                {/* IMAGE */}

                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/infectintoecoli2.png"
                        class="fitimage1"
                    />
                    <p class="fitimage1">
                        Infect into <i>E.coli</i> HB2151 (Second Round)
                    </p>
                </div>

                {/* IMAGE */}
                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/biopanning.png"
                        class="fitimage1"
                    />
                </div>

                {/* IMAGE */}

                {/* PARAGRAPH */}
                <p class="indented1">
                    The graph above shows the results from the first and second
                    round of biopanning, each labeled respectively. In the first
                    round of biopanning, we put the nanobodies at a
                    concentration of 10e11 infectious particles within the
                    sample and received an output of 10e4 infectious particles
                    within the sample when we infected <i>E.coli</i> TG1. We
                    then precipitated the phage. In the second biopanning round,
                    we continued to take the phage that we precipitated by
                    putting it at the same concentration of 10e11. As a result,
                    we got a higher output of 10e8 infectious particles within
                    the sample after infecting it into <i>E.coli</i> HB2151,
                    indicating that we got more specific nanobodies.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />
                {/* IMAGE */}
                <div class="center-horizontal">
                    <img
                        src="https://static.igem.wiki/teams/4314/wiki/elisa.png"
                        class="fitimage1"
                    />
                </div>

                {/* IMAGE */}
                <div class="sizedbox60" />
                {/* PARAGRAPH */}
                <p class="indented1">
                    During the first round of biopanning conducted,{' '}
                    <i>E.coli</i> TG1 was infected in order to obtain a colony,
                    then inoculated at a large scale of 10 mL, and finally
                    spread. For the second round of biopanning, the bacteria was
                    infected into <i>E.coli</i> HB2151 so that the codon became
                    readable.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox20" />
                {/* PARAGRAPH */}
                <p class="indented1">
                    Furthermore, Indirect Enzyme-Linked Immunosorbent Assay
                    (ELISA) was used to test which nanobody binds best with the{' '}
                    <i>R.solanacearum</i> out of the 20 PBS clones. The clones
                    were compared with PBS in order to identify the most
                    specific nanobody. Based on the data graph above, clones
                    number 1, 7, 8, and 17 will be able to bind best with{' '}
                    <i>R.solanacearum</i> . This is because they were the most
                    distinct when compared to the PBS control.
                </p>
                {/* PARAGRAPH */}
                <div class="sizedbox60" />
            </div>
        </div>
    )
}

export default Results
