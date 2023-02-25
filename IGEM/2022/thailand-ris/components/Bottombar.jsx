const Bottombar = () => {
  return (
    <div class="bottombar">
      <div>
        <h2>Contact Us</h2>
        <h3>
          Email:{" "}
          <a
            class="hover-underline-animation"
            href="https://mail.google.com/mail/u/0/?fs=1&to=IGEM2022RIS@gmail.com&tf=cm"
          >
            IGEM2022RIS@gmail.com
          </a>
        </h3>
        <h3>
          Instagram:{" "}
          <a
            class="hover-underline-animation"
            href="https://www.instagram.com/tris.igem22/"
          >
            tris.igem22
          </a>
        </h3>
      </div>
      <img
        src="https://static.igem.wiki/teams/4314/wiki/sponsor.svg"
        class="sponsorimg"
      />
    </div>
  );
};

export default Bottombar;
