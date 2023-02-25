function Engineering({}) {
  return (
    <div>
      <div class="titleScreen1">
        <h1 class="titlescreentext">ENGINEERING</h1>
      </div>
      <div class="body">
        <div class="sizedbox40" />
        <div class="myheader">
          <div class="rounded-container">
            <h1 class="header2">Group1</h1>
          </div>
          <div class="circle1"></div>
        </div>

        <h1 class="header1">Engineering Cycle #1</h1>
        <h2 class="header1">Design</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          In other peptides and enzymes there are additional amino acids
          attached to the head and tail of the structure itself. These amino
          acids are referred to as “g cap” and “poly a tail” which both serve as
          a protection to the peptide strand itself. This same idea is used when
          addressing our natural original bombolitin, thus we decided to make
          three modifications to the bombolitin peptide, and each modification
          has different amino acids attached to the end. We called the three
          modifications LLL, FFF, and WWW. The goal is to be able to express
          these three modifications of bombolitin.
        </p>
        <div class="sizedbox20"></div>
        <p class="indented1">
          The sequence for the original bombolitin is LNLKKILGKIGVMLSHLN. The
          unmodified bombolitin has a net charge of +4, hydrophobic ratio of
          55%, and GRAVY index of 0.58.
        </p>

        <div class="sizedbox20"></div>
        <p class="indented1">
          The sequence for the LLL modification is LNLKKILGKIGVMLSHLNLLL. The
          LLL modification has a net charge of +3.25, hydrophobic ratio of 50%,
          and GRAVY index of 0.91.
        </p>

        <div class="sizedbox20"></div>
        <p class="indented1">
          The sequence for the FFF modification is LNLKKILGKIGVMLSHLNFFF. The
          FFF modification has a net charge of +3.25, hydrophobic ratio of 57%,
          and GRAVY index of 0.90.
        </p>

        <div class="sizedbox20"></div>
        <p class="indented1">
          The sequence for the WWW modification is LNLKKILGKIGVMLSHLNWWW. The
          WWW modification has a net charge of +3.25, hydrophobic ratio of 55%,
          and GRAVY Index of 0.91.
        </p>

        <div class="sizedbox20"></div>

        <h2 class="header1">Build</h2>
        <div class="sizedbox20"></div>
        <p>
          LLL modification
          <br />
          Forward Primers: F_casette_Bombolitin
          <br />
          Reverse Primers: RLm_casette_Bombolitin
        </p>

        <div class="sizedbox20"></div>
        <p>
          FFF modification
          <br />
          Forward Primers: F_casette_Bombolitin
          <br />
          Reverse Primers: RFm_casette_Bombolitin
        </p>

        <div class="sizedbox20"></div>
        <p>
          WWW modification
          <br />
          Forward Primers: F_casette_Bombolitin
          <br />
          Reverse Primers: RWm_casette_Bombolitin
        </p>

        <div class="sizedbox20"></div>
        <p class="indented1">
          These three modifications use the same forward primers, but different
          reverse primers. These modifications all increase the hydrophobicity
          of bombolitin, which will be able to attract negative biomolecules in
          the outer membrane of Ralstonia with its increased positive charge
          from the hydrophobic ratio. Furthermore, the values of the GRAVY Index
          should be near zero, since it means that the peptide will be able to
          bind and have antibacterial activity. Thioredoxin was also able to
          enhance protein expression by decreasing the net charge and
          hydrophobicity. This makes it so that the bombolitin is not toxic to
          the host cell and does not kill it after being produced.
          <div class="center-horizontal">
            <img
              src="https://static.igem.wiki/teams/4314/wiki/engineering-design.png"
              class="fitimage1"
            />
          </div>
        </p>

        <div class="sizedbox20" />
        <h2 class="header1">Test</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          When we expressed each modification, we were only able to express the
          LLL modification successfully. The FFF modification and WWW
          modifications were not able to be expressed.
        </p>
        <div class="sizedbox20"></div>
        <h2 class="header1">Learn</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Although we weren&apos;t able to express all three modifications of
          bombolitin, we registered all three as Parts on the Parts Registry
          Page for other teams to use in the future. The fact that the other two
          modifications were not able to be expressed could be due to multiple
          reasons, such as mistakes with the protocol such as annealing the
          primers at the wrong temperature or using competent cells there were
          too old during heatshock transformation or electroporation.
          Furthermore, we also learned about the crucial role of thioredoxin in
          producing the modified bombolitin since it prevents the competent
          cells from destroying themselves after producing the modified
          bombolitin.
        </p>
        <div class="sizedbox20"></div>

        <h1 class="header1">EngineeringCycle #2</h1>
        <h2 class="header1">Design</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Ligation is a crucial step in Group 1&#39;s work, as it is the segment
          where the recombinant plasmid is created. The process of ligation
          requires a vector, an insert, T4 ligase buffer, T4 ligase, and
          molecular water. Once these have been acquired and mixed together, it
          is to be incubated for 16 hours at 16°C overnight. In this case, the
          vector is the pET-32a, and the insert is the modified bombolitin.
        </p>
        <div class="sizedbox20"></div>

        <h2 class="header1">Build</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          After PCR, restriction enzymes BamHI and SalI are used to purify the
          PCR product. After that, ligation takes place. For this, we used a
          vector to insert a ratio of 6µL:10µL. For ligation we used 6µL of
          pET-32a, 10µL of the bombolitin, 2µL of T4 ligase buffer, 0.25µL of T4
          ligase, and 1.75µL of molecular water. These are then incubated in the
          ligation machine for 16 hours at 16°C.
        </p>

        <div class="sizedbox20" />
        <h2 class="header1">Test</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          After being incubated in the ligation machine for 16 hours, it is run
          through gel electrophoresis to check whether or not the modified
          bombolitin has been successfully inserted into the pET-32a. However,
          we did not get the expected band size.
        </p>
        <div class="sizedbox20" />
        <h2 class="header1">Possible Errors</h2>
        <div class="sizedbox20" />
        <p class="indented1">
          Initially, after we failed for the first few times, we believed that
          the issue may have lied within our different types of transformation.
          This was because the first time we worked through the procedure,
          heatshock transformation showed that the plasmid was present, but as
          we proceeded through electroporation, the plasmid disappeared. This
          made us believe that our electroporation was off, but the second time
          we tried, the plasmid that showed up for heatshock transformation was
          not of the expected band size either. This implied that the plasmid
          was just the empty vector, since the band size was smaller than
          expected, when combined with the insert, the band size should
          increase.
        </p>
        <div class="center-horizontal">
          <img
            src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-03-53-30.png"
            class="fitimage1"
          />
          <p class="fitimage1">
            Size of the bands did not line up with what was expected.
          </p>
        </div>
        <div class="sizedbox20"></div>
        <h2 class="header1">Learn</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          We believe that we did not get the expected band size because the
          vector to insert ratio was too high. Due to this, the ratio of the
          vector to insert was changed to 1:2, where we used 4µL of the vector
          with 8µL of the insert and 3µL of the vector with 6µL of the insert.
          With this new ratio, we were able to get the expected band size
          because we decreased the vector, while keeping the insert the same.
        </p>
        <div class="sizedbox20"></div>

        <div class="myheader">
          <div class="rounded-container">
            <h1 class="header2">Group 2</h1>
          </div>
          <div class="circle1"></div>
        </div>
        <h2 class="header1">Design</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          The Group 2 (Kasetsart University) Lab&#39;s goal is to determine
          which nanobodies bind best with the target bacteria:
          <i> Ralstonia (R.solanacearum)</i>. These nanobodies will be used as a
          target binding antibody in order to fight against bacteria wilt.
          Through biopanning, the values of the various nanobodies that bind
          best with the target bacteria will be identified and measured. The
          results are then checked to ensure whether or not it binds best with
          the target through indirect ELISA.
        </p>
        <p>
          <ul class="some-margin">
            <li>Past pesearch</li>
            <ul class="some-margin">
              <li>
                The process of biopanning consists of several common steps,
                including incubating the target material with the library
                culture, washing unbound bacteria from the material surface,
                placing the washed material with bound library members into
                fresh media, and allowing the bound members to amplify [1].
                There are, however, some aspects of our dry lab that are unique
                and advantageous for our project.
              </li>
            </ul>
          </ul>
        </p>

        <div class="sizedbox20"></div>

        <h2 class="header1">Build</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          In order to determine which nanobody binds best with
          <i> R.solanacearum</i> bacteria to combat bacterial wilt, biopanning
          must be performed twice. <i>R.solanacearum</i> will be prepared as a
          bacteria chain in the form of 96 well plates using a coating buffer
          that was previously prepared to infect <i>E. coli.</i> The
          Phosphate-buffered saline solution that will be used contains 0.05%
          Tween-20, which is less toxic than the alternative, tween-80 and a
          better fit for the experimental procedures. After determining the
          values of the antibodies, 20 random colonies will be selected to test
          for binding capacity through Indirect Enzyme-Linked Immunosorbent
          Assay (ELISA).
        </p>

        <div class="sizedbox20" />
        <div class="center-horizontal">
          <img
            src="https://static.igem.wiki/teams/4314/wiki/screen-shot-2565-10-12-at-04-06-42.png"
            class="fitimage1"
          />
          <p class="fitimage1">Preparing the Coating Buffer</p>
        </div>
        <h2 class="header1">Test</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          Biopanning will be completed by immobilizing each prepared bacteria
          chain onto a well plate with coating buffer. This will be incubated at
          37 degrees celsius overnight. The first round of Biopanning will use
          TG1 <i>E. coli</i>, as it is a suppressor host strain that stops
          protein synthesis when a stop codon is detected. After that, the
          detected protein strain will undergo biopanning again, however, this
          time, using HB2151 <i>E. coli</i> to read the proteins’ codons.
        </p>
        <div class="sizedbox20" />
        <p class="indented1">
          Afterwards, 20 colonies (E.coli HB2151) will be randomly selected to
          determine binding capacity to <i>R.solanacearum</i> by indirect ELISA.
          Finally, the binding capacity will be determined through a color
          indicator where the wells containing the nanobodies will change from
          blue to yellow.
        </p>
        <div class="sizedbox20" />

        <h2 class="header1">Learn</h2>
        <div class="sizedbox20"></div>
        <p class="indented1">
          From the experiment, the members of the Group 2 Lab learned methods of
          biopanning and indirect ELISA as well as the multiple trials needed
          when completing a research lab. Most importantly, they were able to
          determine the nanobody that binds with <i>R.solanacearum</i> from the
          experiment.
        </p>

        <div class="sizedbox40" />
      </div>
    </div>
  );
}

export default Engineering;
