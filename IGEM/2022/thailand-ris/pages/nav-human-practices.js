import Link from "next/link";

const HumanPRacticesNavigation = ({ children }) => {
  return (
    <div>
      <div class="navbg">
        <div class="column body">
          <div class="sizedbox200" />
          <div class="body">
            <h1 class="title">HUMAN PRACTICES</h1>
            <div class="sizedbox40" />
            <div class="center-horizontal">
              <img
                src="https://static.igem.wiki/teams/4314/wiki/icons8-chevron-100.png"
                width={80}
              />
              <div class="sizedbox40" />
              <Link href="/human-practices">
                <h1 class="center-text hover-underline-animation1">
                  INTEGRATED HUMAN PRACTICES
                </h1>
              </Link>
              <div class="sizedbox20" />
              <Link href="/communication">
                <h1 class="center-text hover-underline-animation1">
                  EDUCATIONAL OUTREACH
                </h1>
              </Link>
              <div class="sizedbox20" />
              <Link href="/partnership">
                <h1 class="center-text hover-underline-animation1">
                  PARTNERSHIP
                </h1>
              </Link>
              <div class="sizedbox20" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HumanPRacticesNavigation;
