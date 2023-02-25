// In this file I set everything relating to the meta stuff and font for everything else
import dynamic from 'next/dynamic'
import Head from 'next/head'
// file imports
//
const Main = ({ children, router }: any) => {
    return (
        <div className="font-rockabye bg-white">
            <div>
                <Head>
                    <meta charSet="UTF-8" />
                    <meta
                        name="viewport"
                        content="width=device-width, initial-scale=1.0"
                    />
                    <meta name="description" content="Thailand_RIS" />
                    <meta name="author" content="Thamognya Kodi, Kaka" />
                    <meta
                        name="keywords"
                        content="Thamognya, Kodi, Kaka Links"
                    />
                    <link rel="shortcut icon" href="/assets/pfp.png" />
                    <title>Thailand_RIS</title>
                </Head>
            </div>
            <div>{children}</div>
        </div>
    )
}

export default Main
