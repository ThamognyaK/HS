import Link from "next/link";

const Project = ({}) => {
  return (
    <div>
      <div class="navbg">
        <div class="column body">
          <div class="sizedbox100" />
          <div class="body">
            <h1 class="title">PROJECT</h1>
            <div class="sizedbox40" />
          </div>
          <img
            src="https://static.igem.wiki/teams/4314/wiki/icons8-chevron-100.png"
            width={80}
          />
          <div class="sizedbox40" />
          <Link href="/description">
            <h1 class="center-text hover-underline-animation1">DESCRIPTION</h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/design">
            <h1 class="center-text hover-underline-animation1">DESIGN</h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/engineering">
            <h1 class="center-text hover-underline-animation1">
              ENGINEERING SUCCESS
            </h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/results">
            <h1 class="center-text hover-underline-animation1">RESULTS</h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/implementation">
            <h1 class="center-text hover-underline-animation1">
              PROPOSED IMPLEMENTATION
            </h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/contribution">
            <h1 class="center-text hover-underline-animation1">CONTRIBUTION</h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/notebook">
            <h1 class="center-text hover-underline-animation1">NOTEBOOK</h1>
          </Link>
          <div class="sizedbox20" />
          <Link href="/safety">
            <h1 class="center-text hover-underline-animation1">SAFETY</h1>
          </Link>
          <div class="sizedbox80" />
        </div>
      </div>
    </div>
  );
};

export default Project;
