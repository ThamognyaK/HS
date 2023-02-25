function Design({}) {
  return (
    <div>
      <div class="titleScreen1">
        <h1 class="titlescreentext">DESIGN</h1>
      </div>
      <div class="body">
        <div class="sizedbox40" />
        <h1 class="header1">Research Plans</h1>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Our goal is to find a new peptide that is able to inhibit
          <i> Ralstonia</i> and also genetically modify <i>E.coli</i> plasmid
          (pET-32a) in order to grant it the ability to produce modified
          bombolitin antimicrobial peptide. Hence, the three modifications of
          the bombolitin will be expressed: FFF, WWW, and LLL. We have achieved
          this aim through 3 main steps: Biopanning, Expression, and In-Vitro
          Testing.
        </p>
        <div class="sizedbox20"></div>
        <h1 class="header1">Biopanning</h1>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Biopanning is a technique used to discover new peptides with similar
          behaviors to bombolitin by identifying the values of nanobodies that
          bind best with the target bacteria. The target bacteria,
          <i> R.solanacearum</i>, will be immobilized and prepared as a chain of
          96 well plates containing a Phosphate-buffered saline solution that
          will contain 0.05% Tween-20, which is less toxic and a better fit for
          our procedures. The biopanning process was repeated two times. The
          first was to infect the bacteria into <i>E. coli</i> TG1 and the
          second was to infect <i>E. coli</i> HB2151. After this process, we
          selected the domain nanobody to inhibit <i>R.solanacearum</i> growth.
          Following this, twenty of those colonies will then be randomly picked
          to determine binding capacity to <i>R.solanacearum</i> by Indirect
          Enzyme-Linked Immunosorbent Assay. These nanobodies will be used as a
          target binding antibody in order to fight against bacteria wilt. It
          will show that to obtain antimicrobial peptides, we can use Synbio
          (Group 1) to create novel peptides or use biochemical techniques to
          discover them.
        </p>
        <div class="sizedbox20" />
        <div class="center-horizontal">
          <img
            src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-02-28-20.png"
            class="fitimage1"
          />
        </div>
        <div class="sizedbox20"></div>
        <h1 class="header1">Cloning</h1>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Rather than discovering new peptides, Group 1 lab focuses on
          synthesizing a peptide that has already been discovered and improving
          upon it.
        </p>
        <div class="sizedbox20" />
        <p class="indented1">
          Past research has shown that bombolitin possesses antibacterial
          properties on both gram negative and gram positive bacteria (Qiu,
          2011). However, this is a very long and expensive process, so
          synthetic biology will be used in order to express the bombolitin
          peptide and increase its antimicrobial properties by increasing
          hydrophobicity. In order to increase the efficiency of inhibition
          towards <i>Ralstonia solanacearum</i>, the bombolitin has been
          modified to have increased hydrophobicity by adding three different
          modifications of three additional amino acids at the end of the
          sequence. The three different modifications are LLL, FFF, and WWW.
          These three modifications have the same forward primers
          (F_casette_Bombolitin), which anneal at 63°C, but they have different
          reverse primers. For the LLL modification, the reverse primers are
          R_Lm_casette_Bombolitin, which anneal at 61°C, the FFF modification
          uses the R_Fm_casette_Bombolitin reverse primers, the WWW modification
          uses the R_Wm_casette_Bombolitin reverse primers. These primers will
          be added before doing the PCR, which will amplify the insert.
        </p>
        <div class="sizedbox20" />
        <p class="indented1">
          After the PCR, the plasmid is purified by restriction enzymes BamHI
          and SalI, at which point, ligation will be performed.
        </p>
        <div class="sizedbox20" />
        <p class="indented1">
          From there, via ligation, the vector, pET-32a, is modified and will
          now contain the insert, which is the modified bombolitin. The next
          step is heat shock transformation, where DH5α will take up the
          recombinant plasmid and replicate, then electroporation transformation
          into BL21 to translate the modified bombolitin and create clones for
          protein expression. The competent cell DH5α is then used to duplicate
          more of the combined plasmid, while BL21 is used to induce protein
          expression. After moving combined plasmids into BL21 competent cells,
          protein expression can occur.
        </p>
        <div class="sizedbox20" />

        <h1 class="header1">Protein Expression</h1>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Protein expression is induced by isopropyl β-d-1-thiogalactopyranoside
          and fast protein liquid chromatography. After cloning the BL-21
          competent cells, colonies are selected and incubated. The cells are
          grown until it reaches an OD of 0.6. Then, 0.5mM IPTG, which is a
          lactose inducer, is used to induce protein expression. IPTG interacts
          with the promoter, causing gene expression to occur on regions of the
          lac operator. However, bombolitin is a positively charged hydrophobic
          molecule, this entails that bombolitin will have high affinity to the
          negative protein that is embedded to the outer membrane of cells which
          will rupture the cell membrane itself. Additionally, due to the
          hydrophobic nature of the molecule, bombolitin will be able to
          traverse through the cell membrane and into the cytoplasm, causing
          cell death. Thankfully, the expressed bombolitin will not harm the
          host <i>E .coli</i> competent cell since Thioredoxin, a type of
          protein that solubilizes and reduces the nature and charge of the
          bombolitin, is also expressed. It is then incubated at 37°C before
          being sonicated. Afterwards, there will be only proteins left, which
          will be separated in SDS-Page. FPLC (fast protein liquid
          chromatography) is then performed to purify the proteins. A 5mL
          histrap column is used. After this, there is the process of flow
          through and then elution.
        </p>
        <div class="sizedbox20" />
        <p class="indented1">
          The modified bombolitin has a his-tag, meaning that it will bind to
          the column, while the rest will flow through. The modified bombolitin
          will then be eluted with imidazole in order to get the bombolitin. An
          SDS-page will then be done again to see if the modified bombolitin has
          been purified. If it has been purified, the next step is in-vitro
          testing.
        </p>
        <div class="sizedbox20" />
        <div class="center-horizontal">
          <img
            src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-02-41-13.png"
            class="fitimage1"
          />
        </div>

        <div class="sizedbox20" />
        <h1 class="header1">In-Vitro Testing</h1>
        <div class="sizedbox20" />
        <p class="indented1">
          In-Vitro Testing is a test of the efficiency and effectiveness of the
          modified peptides on different <i>Ralstonia</i> bacteria strains. For
          this, we did an inhibition assay by agar disc diffusion method. There
          are three plates: negative control, positive control, and samples. The
          <i> Ralstonia</i> is grown in a CPG medium, which takes around two
          days. 5mL of <i>Ralstonia</i> is then plated. The disc is placed on
          each plate. After incubating at 30°C for 40 hours, the inhibition zone
          for each plate is measured.
        </p>
        <div class="sizedbox20" />
        <div class="center-horizontal">
          <img
            src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-02-49-29.png"
            class="fitimage1"
          />
        </div>

        <div class="sizedbox40" />
      </div>
    </div>
  );
}

export default Design;
