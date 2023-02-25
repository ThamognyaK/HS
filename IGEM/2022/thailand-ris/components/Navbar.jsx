import Link from "next/link";
import React, { useState } from "react";
import NavItem from "./NavItem";

const MENU_LIST = [
  { text: "HOME", href: "/" },
  { text: "TEAM", href: "/nav-team" },
  { text: "PROJECT", href: "/nav-project" },
  { text: "MODEL", href: "/model" },
  { text: "PARTS", href: "/parts" },
  { text: "HUMAN PRACTICES", href: "/nav-human-practices" },
  {
    text: "JUDGING FORM",
    href: "https://old.igem.org/2022_Judging_Form?id=4314",
  },
];
const Navbar = () => {
  const [navActive, setNavActive] = useState(null);
  const [activeIdx, setActiveIdx] = useState(-1);

  return (
    <header>
      <nav className={`nav`}>
        <Link href={"/"} class="logo">
          <a>
            <h1 class="logo">THAILAND_RIS</h1>
          </a>
        </Link>

        <div
          onClick={() => setNavActive(!navActive)}
          className={`nav__menu-bar`}
        >
          <div></div>
          <div></div>
          <div></div>
        </div>
        <div className={`${navActive ? "active" : ""} nav__menu-list`}>
          {MENU_LIST.map((menu, idx) => (
            <div
              onClick={() => {
                setActiveIdx(idx);
                setNavActive(false);
              }}
              key={menu.text}
            >
              <NavItem active={activeIdx === idx} {...menu} />
            </div>
          ))}
        </div>
      </nav>
    </header>
  );
};

export default Navbar;
